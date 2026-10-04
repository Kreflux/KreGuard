"""Network filter lists in the familiar Adblock Plus syntax.

The same rule format that content blockers use for ads and trackers works
well as an egress denylist for an LLM application: a model that has been
tricked into phoning home usually does it through request catchers, tunnels
and paste sites, and those are exactly what public lists catalogue.

This is an independent implementation written for KreGuard from the public
syntax description. It reads filter lists as data. It contains no code from
any ad blocker.

Supported syntax:

    ! comment                 ignored, as are [Adblock Plus 2.0] headers
    ||example.com^            the domain and every subdomain
    ||example.com/ads/*       a pattern anchored at the host
    |https://example.com/     anchored at the start of the URL
    /track.gif|               anchored at the end of the URL
    ads*banner                substring pattern, ``*`` is a wildcard
    ex^                       ``^`` matches a separator or the end
    @@||example.com/ok^       exception: overrides a matching block rule
    ||example.com^$important  not overridable by exceptions
    example.com               a bare domain, as in domain-only lists
    0.0.0.0 example.com       hosts-file entry, exact host only
    /regex/                   regular expression, only if allow_regex=True

Rules that need browser context (cosmetic filters like ``##.ad``, resource
type options like ``$script`` or ``$domain=``) cannot be evaluated for an
outbound API call. They are handled in the safe direction: a block rule with
unsupported options still blocks, which can only over-block, while an
exception rule with unsupported options is dropped, so it can never widen
what is allowed.

Regular expressions are off by default. A hostile or careless pattern can
make a regex engine run for a very long time, and the guard must not be the
thing that hangs.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Pattern, Union
from urllib.parse import urlsplit

_HOSTNAME = re.compile(r"^(?=.{1,253}$)[a-z0-9_]([a-z0-9_-]{0,61}[a-z0-9_])?(\.[a-z0-9_]([a-z0-9_-]{0,61}[a-z0-9_])?)*$")
_HOSTS_LINE = re.compile(r"^(?:0\.0\.0\.0|127\.0\.0\.1|::1?|::)\s+(\S+)(?:\s+#.*)?$")
_SIMPLE_HOST_RULE = re.compile(r"^\|\|([a-z0-9_.-]+)\^$", re.IGNORECASE)
_OPTION_NAME = re.compile(r"^~?[a-z][a-z0-9_-]*(=.*)?$", re.IGNORECASE)
_COSMETIC = re.compile(r"#[@?$%]?#|#@?\$#|#@?\?#")
_HOSTS_IGNORED = {"localhost", "localhost.localdomain", "local", "broadcasthost", "ip6-localhost", "ip6-loopback", "0.0.0.0"}
_SUPPORTED_OPTIONS = {"important", "match-case"}
_MAX_RULE_LENGTH = 2048

# Request catchers, callback services, tunnels and anonymous paste or file
# drops. A tricked model needs somewhere to send data. These are the usual
# places. Opt in with ``builtin_blocklist``; it is a layer on top of the
# allowlist, which already denies everything it does not name.
DEFAULT_RULES = """\
! KreGuard built-in egress denylist: exfiltration and callback endpoints
! Request catchers and callback services
||webhook.site^
||requestbin.com^
||requestbin.net^
||requestcatcher.com^
||hookbin.com^
||pipedream.net^
||beeceptor.com^
||ptsv2.com^
||postb.in^
||interact.sh^
||oast.pro^
||oast.live^
||oast.site^
||oast.online^
||oast.fun^
||oast.me^
||burpcollaborator.net^
||canarytokens.com^
||dnslog.cn^
||ceye.io^
! Tunnels that expose a local machine
||ngrok.io^
||ngrok.app^
||ngrok-free.app^
||ngrok-free.dev^
||trycloudflare.com^
||localtunnel.me^
||loca.lt^
||serveo.net^
||pagekite.me^
||lhr.life^
! Anonymous paste and file drops
||pastebin.com^
||hastebin.com^
||paste.ee^
||ghostbin.com^
||dpaste.org^
||transfer.sh^
||file.io^
||0x0.st^
||temp.sh^
||anonfiles.com^
||gofile.io^
"""


@dataclass(frozen=True)
class FilterMatch:
    """Outcome of checking one URL against a list."""

    blocked: bool
    rule: Optional[str] = None          # the block rule that matched
    excepted_by: Optional[str] = None   # the exception that cancelled it


@dataclass
class ParseStats:
    block_rules: int = 0
    exception_rules: int = 0
    skipped: int = 0        # comments, cosmetic filters, unusable rules
    approximated: int = 0   # block rules kept with unsupported options
    regex_ignored: int = 0

    def __iadd__(self, other: "ParseStats") -> "ParseStats":
        self.block_rules += other.block_rules
        self.exception_rules += other.exception_rules
        self.skipped += other.skipped
        self.approximated += other.approximated
        self.regex_ignored += other.regex_ignored
        return self


@dataclass(frozen=True)
class _Rule:
    raw: str
    exception: bool
    important: bool


@dataclass(frozen=True)
class _PatternRule(_Rule):
    regex: Pattern[str] = field(default=re.compile(""), compare=False)


def normalize_url_for_matching(url: str) -> Optional[str]:
    """Rebuild a URL as ``scheme://host[:port]/path?query``.

    The host is lowercased and IDNA-encoded, userinfo and fragment are
    dropped. Matching runs on this form so that anchors mean what they say
    and ``https://good.com@evil.com/`` cannot be read as ``good.com``.
    Returns None if the URL cannot be parsed.
    """
    try:
        parts = urlsplit(url.strip())
        host = parts.hostname
        port = parts.port
    except ValueError:
        return None
    if not parts.scheme or not host:
        return None
    host = host.lower().rstrip(".")
    try:
        host = host.encode("idna").decode("ascii")
    except UnicodeError:
        return None
    if ":" in host:  # IPv6 literal
        host = f"[{host}]"
    authority = host if port is None else f"{host}:{port}"
    path = parts.path or "/"
    query = f"?{parts.query}" if parts.query else ""
    return f"{parts.scheme.lower()}://{authority}{path}{query}"


def _split_options(line: str):
    """Split ``pattern$opt,opt`` into (pattern, [options])."""
    idx = line.rfind("$")
    if idx <= 0:
        return line, []
    opts = line[idx + 1 :]
    if not opts:
        return line, []
    names = opts.split(",")
    if not all(_OPTION_NAME.match(n) for n in names):
        return line, []  # the dollar sign belongs to the pattern
    return line[:idx], names


def _translate(pattern: str, flags: int) -> Pattern[str]:
    anchor_host = pattern.startswith("||")
    anchor_start = not anchor_host and pattern.startswith("|")
    if anchor_host:
        pattern = pattern[2:]
    elif anchor_start:
        pattern = pattern[1:]
    anchor_end = pattern.endswith("|")
    if anchor_end:
        pattern = pattern[:-1]

    out: List[str] = []
    for ch in pattern:
        if ch == "*":
            out.append(".*")
        elif ch == "^":
            out.append(r"(?:[^A-Za-z0-9_\-.%]|$)")
        else:
            out.append(re.escape(ch))
    body = "".join(out)

    if anchor_host:
        body = r"^[a-z][a-z0-9+.\-]*://(?:[^/?#:]*\.)?" + body
    elif anchor_start:
        body = "^" + body
    if anchor_end:
        body += "$"
    return re.compile(body, flags)


class FilterList:
    """A set of block and exception rules, queried one URL at a time."""

    def __init__(self, text: Union[str, Iterable[str], None] = None, *, name: str = "", allow_regex: bool = False) -> None:
        self.name = name
        self.allow_regex = allow_regex
        self.stats = ParseStats()
        # Domain rules are indexed by hostname: O(depth) per lookup however
        # long the list. ``exact`` entries come from hosts-file lines.
        self._block_hosts: Dict[str, List[_Rule]] = {}
        self._except_hosts: Dict[str, List[_Rule]] = {}
        self._exact: Dict[str, List[_Rule]] = {}
        self._block_patterns: List[_PatternRule] = []
        self._except_patterns: List[_PatternRule] = []
        if text is not None:
            self.add_text(text if isinstance(text, str) else "\n".join(text))

    # Loading

    @classmethod
    def from_file(cls, path: Union[str, Path], *, allow_regex: bool = False) -> "FilterList":
        p = Path(path)
        fl = cls(name=p.name, allow_regex=allow_regex)
        fl.add_text(p.read_text(encoding="utf-8", errors="replace"))
        return fl

    @classmethod
    def builtin(cls) -> "FilterList":
        return cls(DEFAULT_RULES, name="kreguard-builtin")

    @staticmethod
    def builtin_text() -> str:
        return DEFAULT_RULES

    def add_file(self, path: Union[str, Path]) -> ParseStats:
        return self.add_text(Path(path).read_text(encoding="utf-8", errors="replace"))

    def add_text(self, text: str) -> ParseStats:
        added = ParseStats()
        for line in text.splitlines():
            self._add_line(line, added)
        self.stats += added
        return added

    def __len__(self) -> int:
        hosts = sum(len(v) for v in self._block_hosts.values()) + sum(len(v) for v in self._except_hosts.values())
        exact = sum(len(v) for v in self._exact.values())
        return hosts + exact + len(self._block_patterns) + len(self._except_patterns)

    # Parsing

    def _add_line(self, line: str, stats: ParseStats) -> None:
        line = line.strip()
        if not line or len(line) > _MAX_RULE_LENGTH:
            stats.skipped += 1
            return
        if line.startswith("!") or line.startswith("[") or line.startswith("#"):
            stats.skipped += 1
            return
        if _COSMETIC.search(line):
            stats.skipped += 1
            return

        m = _HOSTS_LINE.match(line)
        if m:
            host = m.group(1).lower().rstrip(".")
            if host in _HOSTS_IGNORED or not _HOSTNAME.match(host):
                stats.skipped += 1
                return
            self._exact.setdefault(host, []).append(_Rule(line, False, False))
            stats.block_rules += 1
            return

        exception = line.startswith("@@")
        body = line[2:] if exception else line
        pattern, options = _split_options(body)
        names = {o.lower().split("=")[0].lstrip("~") for o in options}
        important = "important" in names
        match_case = "match-case" in names
        unsupported = bool(names - _SUPPORTED_OPTIONS)
        if unsupported:
            if exception:
                stats.skipped += 1  # never widen what is allowed
                return
            stats.approximated += 1  # over-blocking is the safe error
        if not pattern:
            stats.skipped += 1
            return

        # Regular expression rule.
        if len(pattern) > 2 and pattern.startswith("/") and pattern.endswith("/"):
            if not self.allow_regex:
                stats.regex_ignored += 1
                return
            try:
                rx = re.compile(pattern[1:-1], 0 if match_case else re.IGNORECASE)
            except re.error:
                stats.skipped += 1
                return
            self._store_pattern(_PatternRule(line, exception, important, rx), exception, stats)
            return

        # Bare domain, as found in domain-only lists.
        if not exception and _HOSTNAME.match(pattern.lower()) and "." in pattern and not options:
            self._store_host(pattern.lower(), _Rule(line, False, False), False, stats)
            return

        # ||host^ is the common case and lives in the index.
        simple = _SIMPLE_HOST_RULE.match(pattern)
        if simple:
            host = simple.group(1).lower().strip(".")
            if _HOSTNAME.match(host):
                self._store_host(host, _Rule(line, exception, important), exception, stats)
                return

        try:
            rx = _translate(pattern, 0 if match_case else re.IGNORECASE)
        except re.error:
            stats.skipped += 1
            return
        self._store_pattern(_PatternRule(line, exception, important, rx), exception, stats)

    def _store_host(self, host: str, rule: _Rule, exception: bool, stats: ParseStats) -> None:
        table = self._except_hosts if exception else self._block_hosts
        table.setdefault(host, []).append(rule)
        if exception:
            stats.exception_rules += 1
        else:
            stats.block_rules += 1

    def _store_pattern(self, rule: _PatternRule, exception: bool, stats: ParseStats) -> None:
        if exception:
            self._except_patterns.append(rule)
            stats.exception_rules += 1
        else:
            self._block_patterns.append(rule)
            stats.block_rules += 1

    # Matching

    def match(self, url: str) -> FilterMatch:
        """Check a URL. Anything that cannot be normalized is not matched here;
        the egress policy rejects unparseable URLs before consulting a list."""
        canon = normalize_url_for_matching(url)
        if canon is None:
            return FilterMatch(False)
        host = urlsplit(canon).hostname or ""
        host = host.strip("[]")

        block = self._first_block(canon, host)
        if block is None:
            return FilterMatch(False)
        if block.important:
            return FilterMatch(True, block.raw)
        exc = self._first_exception(canon, host)
        if exc is not None:
            return FilterMatch(False, block.raw, exc.raw)
        return FilterMatch(True, block.raw)

    @staticmethod
    def _suffixes(host: str):
        labels = host.split(".")
        for i in range(len(labels)):
            yield ".".join(labels[i:])

    def _first_block(self, canon: str, host: str) -> Optional[_Rule]:
        rules = self._exact.get(host)
        if rules:
            return rules[0]
        # An important rule must win over exceptions, so prefer one if present.
        found: Optional[_Rule] = None
        for suffix in self._suffixes(host):
            for rule in self._block_hosts.get(suffix, ()):
                if rule.important:
                    return rule
                found = found or rule
        for prule in self._block_patterns:
            if prule.regex.search(canon):
                if prule.important:
                    return prule
                found = found or prule
        return found

    def _first_exception(self, canon: str, host: str) -> Optional[_Rule]:
        for suffix in self._suffixes(host):
            rules = self._except_hosts.get(suffix)
            if rules:
                return rules[0]
        for prule in self._except_patterns:
            if prule.regex.search(canon):
                return prule
        return None

