"""Load a Guard from a JSON config file.

A policy that lives in a file can be reviewed, diffed and versioned. Unknown
keys are an error: in a security config a typo like ``"alow_tools"`` must not
be silently ignored. Secrets never live here. The server token is read from
an environment variable whose name the config gives.

Example (see examples/kreguard.json):

    {
      "system_prompt_file": "prompt.txt",
      "thresholds": {"flag": 0.35, "block": 0.8},
      "classifier": "adaptive",
      "tools": [
        {"name": "search_orders"},
        {"name": "issue_refund", "confirm": true, "max_calls": 1,
         "args": {"amount": {"type": "number", "max": 500, "required": true}}}
      ],
      "deny_tools": ["shell"],
      "egress": {
        "domains": ["api.acme.com", "*.stripe.com"],
        "ports": [443],
        "builtin_blocklist": true,
        "blocklists": ["lists/extra.txt"]
      },
      "adaptive": {"state_dir": "state", "auto_learn": true},
      "audit": {"path": "audit.jsonl", "include_text": false},
      "server": {"host": "127.0.0.1", "port": 8787, "token_env": "KREGUARD_TOKEN"}
    }
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional, Union

from .adaptive import AdaptiveClassifier, AdaptiveEgress, ModelError
from .audit import AuditLog
from .classifier import LexiconClassifier
from .filterlist import FilterList
from .guard import Guard, GuardConfig
from .permissions import EgressPolicy, ToolPolicy, ToolRule
from .verdict import Verdict


class ConfigError(ValueError):
    """The config is invalid. Raised at load time, never at decision time."""


_TOP_KEYS = {
    "system_prompt", "system_prompt_file", "thresholds", "max_input_chars", "on_error",
    "classifier", "tools", "deny_tools", "egress", "audit", "server", "adaptive",
}
_ADAPT_KEYS = {
    "state_dir", "auto_learn", "autosave_every", "egress", "memory_threshold",
    "auto_per_hour", "min_samples", "flag_z", "block_z",
}
_EGRESS_KEYS = {
    "domains", "schemes", "ports", "allow_ip_literals", "allow_userinfo",
    "max_url_length", "flag_query_secrets", "builtin_blocklist", "blocklists", "allow_regex",
}
_TOOL_KEYS = {"name", "confirm", "max_calls", "description", "args", "strict_args"}
_ARG_KEYS = {"type", "required", "enum", "min", "max", "max_length", "pattern"}
_AUDIT_KEYS = {"path", "include_text"}
_SERVER_KEYS = {"host", "port", "token_env", "max_body_bytes", "playground"}
_TYPES: Dict[str, Callable[[Any], bool]] = {
    "string": lambda v: isinstance(v, str),
    "number": lambda v: isinstance(v, (int, float)) and not isinstance(v, bool),
    "integer": lambda v: isinstance(v, int) and not isinstance(v, bool),
    "boolean": lambda v: isinstance(v, bool),
}


@dataclass
class ServerSettings:
    host: str = "127.0.0.1"
    port: int = 8787
    token_env: Optional[str] = None
    max_body_bytes: int = 1_000_000
    playground: bool = True


@dataclass
class Settings:
    guard: Guard
    server: ServerSettings


def _check_keys(section: str, data: Mapping[str, Any], allowed: set) -> None:
    unknown = sorted(set(data) - allowed)
    if unknown:
        raise ConfigError(f"{section}: unknown key(s) {unknown}; allowed: {sorted(allowed)}")


def _expect(section: str, value: Any, kind: type, what: str) -> Any:
    if not isinstance(value, kind) or (kind is int and isinstance(value, bool)):
        raise ConfigError(f"{section}: expected {what}, got {type(value).__name__}")
    return value


def _str_list(section: str, value: Any) -> List[str]:
    _expect(section, value, list, "a list of strings")
    for item in value:
        _expect(section, item, str, "a list of strings")
    return list(value)


def _arg_validator(tool: str, schema: Mapping[str, Any], strict: bool):
    """Build a validator from declarative per-argument constraints."""
    compiled: Dict[str, Dict[str, Any]] = {}
    for arg, spec in schema.items():
        where = f"tools[{tool}].args.{arg}"
        _expect(where, spec, dict, "an object")
        _check_keys(where, spec, _ARG_KEYS)
        entry = dict(spec)
        if "type" in entry and entry["type"] not in _TYPES:
            raise ConfigError(f"{where}: type must be one of {sorted(_TYPES)}")
        if "pattern" in entry:
            try:
                entry["pattern"] = re.compile(_expect(where, entry["pattern"], str, "a regex string"))
            except re.error as exc:
                raise ConfigError(f"{where}: bad pattern: {exc}") from exc
        if "enum" in entry:
            _expect(where, entry["enum"], list, "a list")
        compiled[arg] = entry

    def validate(args: Mapping[str, Any]):
        if strict:
            extra = sorted(set(args) - set(compiled))
            if extra:
                return f"unexpected argument(s): {extra}"
        for name, spec in compiled.items():
            if name not in args:
                if spec.get("required"):
                    return f"missing required argument {name!r}"
                continue
            value = args[name]
            t = spec.get("type")
            if t and not _TYPES[t](value):
                return f"argument {name!r} must be {t}"
            if "enum" in spec and value not in spec["enum"]:
                return f"argument {name!r} not in allowed values"
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                if "min" in spec and value < spec["min"]:
                    return f"argument {name!r} below minimum {spec['min']}"
                if "max" in spec and value > spec["max"]:
                    return f"argument {name!r} above maximum {spec['max']}"
            if isinstance(value, str):
                if "max_length" in spec and len(value) > spec["max_length"]:
                    return f"argument {name!r} longer than {spec['max_length']}"
                if "pattern" in spec and not spec["pattern"].fullmatch(value):
                    return f"argument {name!r} does not match the required pattern"
            if ("min" in spec or "max" in spec) and not isinstance(value, (int, float)):
                return f"argument {name!r} must be numeric"
        return True

    return validate


def _build_tools(data: Mapping[str, Any]) -> ToolPolicy:
    policy = ToolPolicy()
    for i, raw in enumerate(_expect("tools", data.get("tools", []), list, "a list of tool objects")):
        where = f"tools[{i}]"
        _expect(where, raw, dict, "an object")
        _check_keys(where, raw, _TOOL_KEYS)
        name = _expect(where, raw.get("name"), str, "a tool name")
        if not name:
            raise ConfigError(f"{where}: name is empty")
        validate = None
        if "args" in raw:
            validate = _arg_validator(name, _expect(where, raw["args"], dict, "an object"), bool(raw.get("strict_args", False)))
        elif raw.get("strict_args"):
            validate = _arg_validator(name, {}, True)
        max_calls = raw.get("max_calls")
        if max_calls is not None and (_expect(where, max_calls, int, "an integer") < 0):
            raise ConfigError(f"{where}: max_calls must be >= 0")
        policy.allow(ToolRule(
            name,
            validate=validate,
            confirm=bool(raw.get("confirm", False)),
            max_calls=max_calls,
            description=str(raw.get("description", "")),
        ))
    policy.deny(*_str_list("deny_tools", data.get("deny_tools", [])))
    return policy


def _build_egress(data: Mapping[str, Any], base: Path, learner: Optional[AdaptiveEgress] = None) -> EgressPolicy:
    _check_keys("egress", data, _EGRESS_KEYS)
    kwargs: Dict[str, Any] = {"domains": set(_str_list("egress.domains", data.get("domains", [])))}
    if "schemes" in data:
        kwargs["schemes"] = set(_str_list("egress.schemes", data["schemes"]))
    if "ports" in data:
        ports = _expect("egress.ports", data["ports"], list, "a list of ports")
        if not all(isinstance(p, int) and not isinstance(p, bool) and 0 < p < 65536 for p in ports):
            raise ConfigError("egress.ports: every port must be an integer from 1 to 65535")
        kwargs["ports"] = set(ports)
    for key in ("allow_ip_literals", "allow_userinfo", "flag_query_secrets"):
        if key in data:
            kwargs[key] = bool(_expect(f"egress.{key}", data[key], bool, "true or false"))
    if "max_url_length" in data:
        kwargs["max_url_length"] = _expect("egress.max_url_length", data["max_url_length"], int, "an integer")

    allow_regex = bool(data.get("allow_regex", False))
    use_builtin = bool(data.get("builtin_blocklist", True))
    paths = _str_list("egress.blocklists", data.get("blocklists", []))
    if use_builtin or paths:
        blocklist = FilterList(name="config", allow_regex=allow_regex)
        if use_builtin:
            blocklist.add_text(FilterList.builtin_text())
        for rel in paths:
            path = (base / rel) if not Path(rel).is_absolute() else Path(rel)
            try:
                blocklist.add_file(path)
            except OSError as exc:
                raise ConfigError(f"egress.blocklists: cannot read {path}: {exc}") from exc
        kwargs["blocklist"] = blocklist
    if learner is not None:
        kwargs["learner"] = learner
    return EgressPolicy(**kwargs)


def guard_from_dict(data: Mapping[str, Any], base_dir: Union[str, Path] = ".") -> Settings:
    base = Path(base_dir)
    _expect("config", data, dict, "an object")
    _check_keys("config", data, _TOP_KEYS)

    gc: Dict[str, Any] = {}
    if "thresholds" in data:
        th = _expect("thresholds", data["thresholds"], dict, "an object")
        _check_keys("thresholds", th, {"flag", "block"})
        if "flag" in th:
            gc["flag_threshold"] = float(th["flag"])
        if "block" in th:
            gc["block_threshold"] = float(th["block"])
    if "max_input_chars" in data:
        gc["max_input_chars"] = _expect("max_input_chars", data["max_input_chars"], int, "an integer")
    if "on_error" in data:
        try:
            gc["on_error"] = Verdict(data["on_error"])
        except ValueError as exc:
            raise ConfigError("on_error must be 'flag' or 'block'") from exc
    try:
        config = GuardConfig(**gc)
    except ValueError as exc:
        raise ConfigError(str(exc)) from exc

    prompt: Optional[str] = None
    if "system_prompt" in data and "system_prompt_file" in data:
        raise ConfigError("give system_prompt or system_prompt_file, not both")
    if "system_prompt" in data:
        prompt = _expect("system_prompt", data["system_prompt"], str, "a string")
    elif "system_prompt_file" in data:
        path = base / _expect("system_prompt_file", data["system_prompt_file"], str, "a path")
        try:
            prompt = path.read_text(encoding="utf-8")
        except OSError as exc:
            raise ConfigError(f"system_prompt_file: cannot read {path}: {exc}") from exc

    classifier_name = data.get("classifier", "adaptive")
    if classifier_name not in ("adaptive", "lexicon", "none", None):
        raise ConfigError("classifier must be 'adaptive', 'lexicon' or 'none'")

    adaptive = _expect("adaptive", data.get("adaptive", {}), dict, "an object")
    _check_keys("adaptive", adaptive, _ADAPT_KEYS)
    state_dir: Optional[Path] = None
    if "state_dir" in adaptive:
        sd = Path(_expect("adaptive.state_dir", adaptive["state_dir"], str, "a directory path"))
        state_dir = sd if sd.is_absolute() else base / sd
    if "auto_learn" in adaptive:
        config.auto_learn = bool(_expect("adaptive.auto_learn", adaptive["auto_learn"], bool, "true or false"))
    autosave = _expect("adaptive.autosave_every", adaptive.get("autosave_every", 25), int, "an integer")

    classifier = None
    egress_learner: Optional[AdaptiveEgress] = None
    try:
        if classifier_name == "lexicon":
            classifier = LexiconClassifier()
        elif classifier_name == "adaptive":
            ckw: Dict[str, Any] = {"autosave_every": autosave}
            if "memory_threshold" in adaptive:
                ckw["memory_threshold"] = float(adaptive["memory_threshold"])
            if "auto_per_hour" in adaptive:
                ckw["auto_per_hour"] = _expect("adaptive.auto_per_hour", adaptive["auto_per_hour"], int, "an integer")
            classifier = AdaptiveClassifier(path=(state_dir / "text-model.json") if state_dir else None, **ckw)
        if bool(adaptive.get("egress", True)):
            ekw: Dict[str, Any] = {"autosave_every": autosave}
            for key in ("min_samples", "flag_z", "block_z"):
                if key in adaptive:
                    ekw[key] = adaptive[key]
            # An operator who turns the built-in denylist off does not want its
            # hosts smuggled back in through the learner's seed.
            ekw["seed_malicious"] = bool(_expect("egress", data.get("egress", {}), dict, "an object").get("builtin_blocklist", True))
            egress_learner = AdaptiveEgress(path=(state_dir / "egress-model.json") if state_dir else None, **ekw)
    except ModelError as exc:
        raise ConfigError(f"adaptive model could not be loaded, refusing to start: {exc}") from exc
    except ValueError as exc:
        raise ConfigError(f"adaptive: {exc}") from exc

    audit: Optional[AuditLog] = None
    if "audit" in data:
        a = _expect("audit", data["audit"], dict, "an object")
        _check_keys("audit", a, _AUDIT_KEYS)
        apath = _expect("audit.path", a.get("path"), str, "a path")
        try:
            audit = AuditLog((base / apath) if not Path(apath).is_absolute() else apath, include_text=bool(a.get("include_text", False)))
        except OSError as exc:
            raise ConfigError(f"audit.path: cannot open {apath}: {exc}") from exc

    server = ServerSettings()
    if "server" in data:
        s = _expect("server", data["server"], dict, "an object")
        _check_keys("server", s, _SERVER_KEYS)
        if "host" in s:
            server.host = _expect("server.host", s["host"], str, "a host string")
        if "port" in s:
            server.port = _expect("server.port", s["port"], int, "an integer")
            if not 0 <= server.port < 65536:
                raise ConfigError("server.port must be between 0 and 65535")
        if "token_env" in s:
            server.token_env = _expect("server.token_env", s["token_env"], str, "an environment variable name")
        if "max_body_bytes" in s:
            server.max_body_bytes = _expect("server.max_body_bytes", s["max_body_bytes"], int, "an integer")
        if "playground" in s:
            server.playground = bool(_expect("server.playground", s["playground"], bool, "true or false"))

    guard = Guard(
        config=config,
        classifier=classifier,
        tool_policy=_build_tools(data),
        egress_policy=_build_egress(data.get("egress", {}), base, egress_learner),
        system_prompt=prompt,
        audit=audit,
    )
    return Settings(guard=guard, server=server)


def load_settings(path: Union[str, Path]) -> Settings:
    p = Path(path)
    try:
        raw = json.loads(p.read_text(encoding="utf-8"))
    except OSError as exc:
        raise ConfigError(f"cannot read config {p}: {exc}") from exc
    except json.JSONDecodeError as exc:
        raise ConfigError(f"config {p} is not valid JSON: {exc}") from exc
    return guard_from_dict(raw, p.parent)
