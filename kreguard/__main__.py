"""Command line entry point.

    python -m kreguard input "ignore all previous instructions"
    echo "..." | python -m kreguard input -
    python -m kreguard output --system-prompt prompt.txt reply.txt
    python -m kreguard egress https://api.example.com/v1 --allow api.example.com
    python -m kreguard egress https://example.com --allow '*.com' --blocklist list.txt
    python -m kreguard serve --config kreguard.json

``--config`` loads a policy file for any command. Exit code is 0 for allow,
1 for flag, 2 for block, 3 for a configuration error.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Optional

from . import __version__
from .classifier import LexiconClassifier
from .config import ConfigError, Settings, load_settings
from .filterlist import FilterList
from .guard import Guard
from .permissions import EgressPolicy
from .verdict import Verdict

_EXIT = {Verdict.ALLOW: 0, Verdict.FLAG: 1, Verdict.BLOCK: 2}
_USAGE_ERROR = 3


def _read(arg: str) -> str:
    if arg == "-":
        return sys.stdin.read()
    p = Path(arg)
    if p.is_file():
        return p.read_text(encoding="utf-8", errors="replace")
    return arg


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="kreguard", description="KreGuard: guardrails for LLM apps.")
    parser.add_argument("--version", action="version", version=f"kreguard {__version__}")
    sub = parser.add_subparsers(dest="cmd", required=True)

    def add_config(p: argparse.ArgumentParser) -> None:
        p.add_argument("--config", help="JSON policy file (see examples/kreguard.json)")

    p_in = sub.add_parser("input", help="scan an input for injection or jailbreak attempts")
    p_in.add_argument("text", help="text, a file path, or - for stdin")
    p_in.add_argument("--no-classifier", action="store_true", help="patterns only")
    p_in.add_argument("--json", action="store_true")
    add_config(p_in)

    p_out = sub.add_parser("output", help="scan a model output for leakage and credentials")
    p_out.add_argument("text", help="text, a file path, or - for stdin")
    p_out.add_argument("--system-prompt", help="text or file path of the system prompt")
    p_out.add_argument("--json", action="store_true")
    add_config(p_out)

    p_eg = sub.add_parser("egress", help="check a URL against the allowlist and filter lists")
    p_eg.add_argument("url")
    p_eg.add_argument("--allow", action="append", default=[], help="allowed domain (repeatable)")
    p_eg.add_argument("--blocklist", action="append", default=[], metavar="FILE", help="Adblock-syntax filter list (repeatable)")
    p_eg.add_argument("--builtin-blocklist", action="store_true", help="also apply the built-in exfiltration denylist")
    p_eg.add_argument("--json", action="store_true")
    add_config(p_eg)

    p_srv = sub.add_parser("serve", help="run the HTTP service and playground")
    p_srv.add_argument("--host", help="bind address (default 127.0.0.1)")
    p_srv.add_argument("--port", type=int, help="port (default 8787)")
    p_srv.add_argument("--token-env", metavar="NAME", help="environment variable that holds the API token")
    p_srv.add_argument("--no-playground", action="store_true", help="do not serve the web page at /")
    add_config(p_srv)
    return parser


def _settings(args: argparse.Namespace) -> Optional[Settings]:
    return load_settings(args.config) if getattr(args, "config", None) else None


def main(argv=None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    try:
        return _run(args)
    except ConfigError as exc:
        print(f"kreguard: {exc}", file=sys.stderr)
        return _USAGE_ERROR


def _run(args: argparse.Namespace) -> int:
    settings = _settings(args)

    if args.cmd == "serve":
        from .server import serve

        guard = settings.guard if settings else Guard(classifier=LexiconClassifier())
        srv = settings.server if settings else None
        host = args.host or (srv.host if srv else "127.0.0.1")
        port = args.port if args.port is not None else (srv.port if srv else 8787)
        token_env = args.token_env or (srv.token_env if srv else None)
        token = None
        if token_env:
            token = os.environ.get(token_env)
            if not token:
                raise ConfigError(f"environment variable {token_env} is empty or not set")
        playground = not args.no_playground and (srv.playground if srv else True)
        try:
            serve(guard, host, port, token, srv.max_body_bytes if srv else 1_000_000, playground)
        except ValueError as exc:
            raise ConfigError(str(exc)) from exc
        return 0

    if args.cmd == "input":
        if settings:
            guard = settings.guard
            if args.no_classifier:
                guard.classifier = None
        else:
            guard = Guard(classifier=None if args.no_classifier else LexiconClassifier())
        decision = guard.check_input(_read(args.text))
    elif args.cmd == "output":
        if settings:
            guard = settings.guard
            decision = guard.check_output(_read(args.text), _read(args.system_prompt) if args.system_prompt else None)
        else:
            guard = Guard(system_prompt=_read(args.system_prompt) if args.system_prompt else None)
            decision = guard.check_output(_read(args.text))
    else:
        if settings:
            guard = settings.guard
            policy = guard.egress_policy
            policy.allow(*args.allow)
        else:
            policy = EgressPolicy(domains=set(args.allow))
            guard = Guard(egress_policy=policy)
        if args.blocklist or args.builtin_blocklist:
            if policy.blocklist is None:
                policy.blocklist = FilterList(name="cli")
            try:
                for path in args.blocklist:
                    policy.blocklist.add_file(path)
            except OSError as exc:
                raise ConfigError(f"cannot read blocklist: {exc}") from exc
            if args.builtin_blocklist:
                policy.blocklist.add_text(FilterList.builtin_text())
        decision = guard.authorize_egress(args.url)

    if args.json:
        payload = decision.as_dict()
        redacted = getattr(decision, "redacted", None)
        if redacted is not None:
            payload["redacted"] = redacted
        print(json.dumps(payload, indent=2))
    else:
        print(f"{decision.verdict.value.upper()}  score={decision.score:.2f}")
        for f in decision.findings:
            print(f"  [{f.source}] {f.rule}: {f.detail}")
        redacted = getattr(decision, "redacted", None)
        if redacted is not None and args.cmd == "output" and decision.verdict is not Verdict.ALLOW:
            print("  redacted:", redacted)
    return _EXIT[decision.verdict]


if __name__ == "__main__":
    sys.exit(main())
