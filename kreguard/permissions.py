"""Permission gates: the part that actually enforces.

Everything else in KreGuard reads text and guesses. This module does not
guess. A tool is callable or it is not. A domain is reachable or it is not.
These decisions are made from static policy, independent of what the model
said or how convincing the attacker was. When the text defenses fail, and
they will, the blast radius is whatever these gates permit.

Defaults are deny. An empty policy allows nothing.
"""

from __future__ import annotations

import ipaddress
import re
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterable, Mapping, Optional, Set, Union
from urllib.parse import urlsplit

from .verdict import Decision, Finding, Verdict

ArgValidator = Callable[[Mapping[str, Any]], Union[bool, str, None]]


@dataclass
class ToolRule:
    """Per-tool policy.

    ``validate`` receives the argument mapping and returns True (or None) to
    allow, False to block, or a string describing why it should be blocked.
    ``confirm`` marks tools that need a human in the loop: they return FLAG
    rather than ALLOW so the caller can pause.
    """

    name: str
    validate: Optional[ArgValidator] = None
    confirm: bool = False
    max_calls: Optional[int] = None
    description: str = ""


class ToolPolicy:
    """Allowlist of tools with optional argument validation and call budgets."""

    def __init__(self, rules: Iterable[ToolRule] = (), denied: Iterable[str] = ()) -> None:
        self._rules: Dict[str, ToolRule] = {}
        for rule in rules:
            self.allow(rule)
        self._denied: Set[str] = set(denied)
        self._calls: Dict[str, int] = {}

    def allow(self, rule: Union[ToolRule, str], **kwargs: Any) -> "ToolPolicy":
        if isinstance(rule, str):
            rule = ToolRule(rule, **kwargs)
        self._rules[rule.name] = rule
        return self

    def deny(self, *names: str) -> "ToolPolicy":
        self._denied.update(names)
        return self

    def reset_budgets(self) -> None:
        self._calls.clear()

    @property
    def allowed_tools(self) -> Set[str]:
        return set(self._rules) - self._denied

    def authorize(self, tool: str, args: Optional[Mapping[str, Any]] = None) -> Decision:
        args = args or {}
        decision = Decision(verdict=Verdict.ALLOW, stage="tools")
        if not isinstance(tool, str) or not tool:
            decision.add(Finding("tools", "invalid_tool_name", Verdict.BLOCK, repr(tool), 1.0))
            return decision
        if tool in self._denied:
            decision.add(Finding("tools", "denied", Verdict.BLOCK, f"{tool} is on the deny list", 1.0))
            return decision
        rule = self._rules.get(tool)
        if rule is None:
            decision.add(Finding("tools", "not_allowlisted", Verdict.BLOCK, f"{tool} is not in the allowlist", 1.0))
            return decision

        if rule.max_calls is not None:
            used = self._calls.get(tool, 0)
            if used >= rule.max_calls:
                decision.add(Finding("tools", "budget_exhausted", Verdict.BLOCK, f"{tool} called {used}/{rule.max_calls} times", 1.0))
                return decision

        if rule.validate is not None:
            try:
                outcome = rule.validate(args)
            except Exception as exc:  # noqa: BLE001 - validator failure is a block
                decision.add(Finding("tools", "validator_error", Verdict.BLOCK, f"{type(exc).__name__}: {exc}", 1.0))
                return decision
            if outcome is False:
                decision.add(Finding("tools", "args_rejected", Verdict.BLOCK, f"{tool} arguments rejected", 1.0))
                return decision
            if isinstance(outcome, str):
                decision.add(Finding("tools", "args_rejected", Verdict.BLOCK, outcome, 1.0))
                return decision
            if outcome is not True and outcome is not None:
                decision.add(Finding("tools", "validator_error", Verdict.BLOCK, f"validator returned {outcome!r}", 1.0))
                return decision

        self._calls[tool] = self._calls.get(tool, 0) + 1
        if rule.confirm:
            decision.add(Finding("tools", "needs_confirmation", Verdict.FLAG, f"{tool} requires human confirmation", 0.5))
        return decision


_METADATA_HOSTS = {
    "metadata.google.internal",
    "metadata",
    "instance-data",
}
_HOSTNAME = re.compile(r"^(?=.{1,253}$)(?!-)[a-z0-9-]{1,63}(?<!-)(\.(?!-)[a-z0-9-]{1,63}(?<!-))*\.?$")


@dataclass
class EgressPolicy:
    """Allowlist of destinations the application may contact.

    Domains match exactly or, with a leading ``*.``, any subdomain. Nothing
    else is reachable. Private, loopback, link-local and cloud metadata
    addresses are blocked even if an allowlist entry would otherwise cover
    them, because a resolver can be pointed anywhere.
    """

    domains: Set[str] = field(default_factory=set)
    schemes: Set[str] = field(default_factory=lambda: {"https"})
    ports: Optional[Set[int]] = None
    allow_ip_literals: bool = False
    allow_userinfo: bool = False
    max_url_length: int = 2048
    flag_query_secrets: bool = True

    def __post_init__(self) -> None:
        self.domains = {self._canon(d) for d in self.domains}
        self.schemes = {s.lower() for s in self.schemes}

    @staticmethod
    def _canon(domain: str) -> str:
        d = domain.strip().lower().rstrip(".")
        return d

    def allow(self, *domains: str) -> "EgressPolicy":
        for d in domains:
            self.domains.add(self._canon(d))
        return self

    def _domain_allowed(self, host: str) -> bool:
        for entry in self.domains:
            if entry.startswith("*."):
                suffix = entry[1:]  # ".example.com"
                if host.endswith(suffix) and host != suffix[1:]:
                    return True
                if host == suffix[1:]:
                    # "*.example.com" also covers the apex, by convention here.
                    return True
            elif host == entry:
                return True
        return False

    def authorize(self, url: str) -> Decision:
        decision = Decision(verdict=Verdict.ALLOW, stage="egress")

        def block(rule: str, detail: str) -> Decision:
            decision.add(Finding("egress", rule, Verdict.BLOCK, detail, 1.0))
            return decision

        if not isinstance(url, str) or not url.strip():
            return block("invalid_url", repr(url))
        if len(url) > self.max_url_length:
            return block("url_too_long", f"{len(url)} chars")
        if any(ord(c) < 0x20 or c in "\\" for c in url):
            return block("control_chars", "url contains control or escape characters")

        try:
            parts = urlsplit(url.strip())
        except ValueError as exc:
            return block("unparseable", str(exc))

        scheme = (parts.scheme or "").lower()
        if scheme not in self.schemes:
            return block("scheme_denied", f"{scheme or '<none>'} not in {sorted(self.schemes)}")

        if parts.username is not None or parts.password is not None:
            if not self.allow_userinfo:
                return block("userinfo_in_url", "credentials embedded in URL")

        try:
            host = parts.hostname
            port = parts.port
        except ValueError as exc:
            return block("invalid_authority", str(exc))
        if not host:
            return block("missing_host", url)
        host = host.lower().rstrip(".")
        try:
            host = host.encode("idna").decode("ascii")
        except UnicodeError:
            return block("invalid_hostname", "IDNA encoding failed")

        if host in _METADATA_HOSTS:
            return block("metadata_endpoint", host)

        ip: Optional[Union[ipaddress.IPv4Address, ipaddress.IPv6Address]] = None
        try:
            ip = ipaddress.ip_address(host.strip("[]"))
        except ValueError:
            ip = None

        if ip is not None:
            if (
                ip.is_private
                or ip.is_loopback
                or ip.is_link_local
                or ip.is_multicast
                or ip.is_reserved
                or ip.is_unspecified
                or str(ip) == "169.254.169.254"
            ):
                return block("private_address", str(ip))
            if not self.allow_ip_literals:
                return block("ip_literal", str(ip))
            if not self._domain_allowed(str(ip)):
                return block("not_allowlisted", str(ip))
        else:
            if not _HOSTNAME.match(host):
                return block("invalid_hostname", host)
            if host in ("localhost",) or host.endswith(".localhost") or host.endswith(".local") or host.endswith(".internal"):
                return block("private_hostname", host)
            if not self._domain_allowed(host):
                return block("not_allowlisted", host)

        if self.ports is not None and port is not None and port not in self.ports:
            return block("port_denied", str(port))

        if self.flag_query_secrets and parts.query:
            if re.search(r"(?i)(token|key|secret|password|passwd|auth|session|credential|cookie)=", parts.query):
                decision.add(Finding("egress", "secret_in_query", Verdict.FLAG, "query string carries a secret-looking parameter", 0.6))
            if re.search(r"[A-Za-z0-9_\-]{40,}", parts.query):
                decision.add(Finding("egress", "long_opaque_query", Verdict.FLAG, "query string carries a long opaque value", 0.4))

        return decision
