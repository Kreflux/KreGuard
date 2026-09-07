"""Output scanning: prompt leakage and credential shapes.

Input filtering fails eventually. Output scanning is the second chance.
Two questions are asked of every model reply:

1. Does it reproduce the system prompt? Compared by shingles over normalized
   words, so paraphrase-resistant enough to catch "repeat everything above"
   while ignoring a shared word or two.
2. Does it contain something shaped like a credential? Cloud keys, tokens,
   private keys, JWTs, connection strings. These are blocked and redacted
   regardless of where they came from.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Pattern, Sequence, Set, Tuple

from .normalize import normalize
from .verdict import Decision, Finding, Verdict


@dataclass(frozen=True)
class CredentialRule:
    name: str
    pattern: Pattern[str]
    verdict: Verdict = Verdict.BLOCK
    description: str = ""


def _c(name: str, regex: str, verdict: Verdict = Verdict.BLOCK, description: str = "") -> CredentialRule:
    return CredentialRule(name, re.compile(regex), verdict, description)


CREDENTIAL_RULES: List[CredentialRule] = [
    _c("aws_access_key", r"\b(?:AKIA|ASIA|AGPA|AIDA|AROA|ANPA|ANVA)[A-Z0-9]{16}\b"),
    _c("aws_secret_key", r"(?i)\baws[_\-\s]?(?:secret|secret_access)[_\-\s]?(?:key)?\b[\s:=\"']{0,6}[A-Za-z0-9/+=]{40}\b"),
    _c("github_token", r"\b(?:ghp|gho|ghu|ghs|ghr)_[A-Za-z0-9]{36,255}\b"),
    _c("github_fine_grained", r"\bgithub_pat_[A-Za-z0-9_]{82}\b"),
    _c("gitlab_token", r"\bglpat-[A-Za-z0-9\-_]{20}\b"),
    _c("slack_token", r"\bxox[abprs]-[A-Za-z0-9\-]{10,}\b"),
    _c("slack_webhook", r"https://hooks\.slack\.com/services/T[A-Z0-9]+/B[A-Z0-9]+/[A-Za-z0-9]+"),
    _c("stripe_key", r"\b(?:sk|rk)_(?:live|test)_[A-Za-z0-9]{16,}\b"),
    _c("openai_key", r"\bsk-(?:proj-)?[A-Za-z0-9_\-]{20,}\b"),
    _c("anthropic_key", r"\bsk-ant-[A-Za-z0-9_\-]{20,}\b"),
    _c("google_api_key", r"\bAIza[0-9A-Za-z\-_]{35}\b"),
    _c("google_oauth", r"\b[0-9]+-[A-Za-z0-9_]{32}\.apps\.googleusercontent\.com\b"),
    _c("sendgrid_key", r"\bSG\.[A-Za-z0-9_\-]{22}\.[A-Za-z0-9_\-]{43}\b"),
    _c("twilio_key", r"\bSK[0-9a-fA-F]{32}\b"),
    _c("npm_token", r"\bnpm_[A-Za-z0-9]{36}\b"),
    _c("pypi_token", r"\bpypi-AgEIcHlwaS5vcmc[A-Za-z0-9\-_]{50,}\b"),
    _c("vercel_token", r"\bvcp_[A-Za-z0-9]{24,}\b"),
    _c("jwt", r"\beyJ[A-Za-z0-9_\-]{10,}\.eyJ[A-Za-z0-9_\-]{10,}\.[A-Za-z0-9_\-]{10,}\b"),
    _c("private_key_block", r"-----BEGIN (?:RSA |EC |DSA |OPENSSH |PGP |ENCRYPTED )?PRIVATE KEY(?: BLOCK)?-----"),
    _c("connection_string", r"\b(?:postgres(?:ql)?|mysql|mongodb(?:\+srv)?|redis|amqp|mssql)://[^\s:@/]+:[^\s@/]+@[^\s/]+"),
    _c("basic_auth_url", r"\bhttps?://[^\s:@/]+:[^\s@/]{4,}@[^\s/]+"),
    _c(
        "generic_secret_assignment",
        r"(?i)\b(?:api[_\-]?key|secret[_\-]?key|access[_\-]?token|auth[_\-]?token|client[_\-]?secret|private[_\-]?key|password|passwd|pwd)\b\s*[:=]\s*[\"']?([A-Za-z0-9_\-/+=.]{16,})[\"']?",
        Verdict.FLAG,
        "Something assigned to a secret-looking name. Flagged, not blocked: could be a placeholder.",
    ),
    _c("bearer_header", r"(?i)\bauthorization\s*:\s*(?:bearer|basic|token)\s+[A-Za-z0-9_\-.=/+]{20,}", Verdict.BLOCK),
]

_TOKEN_LIKE = re.compile(r"\b[A-Za-z0-9_\-/+=]{32,}\b")
_LEAK_PHRASES = re.compile(
    r"\b(my|the) (system|initial|hidden|original) (prompt|instructions?) (is|are|was|were|says?|reads?)\b|\bi (was|am) (told|instructed|configured|programmed) to\b",
    re.IGNORECASE,
)


def _shannon_entropy(s: str) -> float:
    if not s:
        return 0.0
    counts: Dict[str, int] = {}
    for ch in s:
        counts[ch] = counts.get(ch, 0) + 1
    n = len(s)
    return -sum((c / n) * math.log2(c / n) for c in counts.values())


def _words(text: str) -> List[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


def _shingles(words: Sequence[str], k: int) -> Set[Tuple[str, ...]]:
    if len(words) < k:
        return {tuple(words)} if words else set()
    return {tuple(words[i : i + k]) for i in range(len(words) - k + 1)}


@dataclass
class OutputDecision(Decision):
    """A Decision plus a redacted copy of the output safe to pass on."""

    redacted: str = ""
    leakage_ratio: float = 0.0
    credential_hits: List[str] = field(default_factory=list)

    def as_dict(self) -> dict:
        d = super().as_dict()
        d["leakage_ratio"] = round(self.leakage_ratio, 4)
        d["credential_hits"] = list(self.credential_hits)
        return d


class OutputScanner:
    """Scan model output before it reaches the user or a downstream system."""

    def __init__(
        self,
        system_prompt: Optional[str] = None,
        shingle_size: int = 6,
        leak_flag_ratio: float = 0.08,
        leak_block_ratio: float = 0.25,
        min_leak_shingles: int = 2,
        credential_rules: Optional[Sequence[CredentialRule]] = None,
        entropy_threshold: float = 4.5,
        redact_with: str = "[REDACTED]",
    ) -> None:
        if not (0.0 < leak_flag_ratio < leak_block_ratio <= 1.0):
            raise ValueError("leak ratios must satisfy 0 < flag < block <= 1")
        self.shingle_size = shingle_size
        self.leak_flag_ratio = leak_flag_ratio
        self.leak_block_ratio = leak_block_ratio
        self.min_leak_shingles = min_leak_shingles
        self.credential_rules = list(CREDENTIAL_RULES if credential_rules is None else credential_rules)
        self.entropy_threshold = entropy_threshold
        self.redact_with = redact_with
        self._system_shingles: Set[Tuple[str, ...]] = set()
        self._system_text = ""
        if system_prompt:
            self.set_system_prompt(system_prompt)

    def set_system_prompt(self, system_prompt: str) -> None:
        canonical = normalize(system_prompt).canonical
        self._system_text = canonical.lower()
        self._system_shingles = _shingles(_words(canonical), self.shingle_size)

    # Prompt leakage

    def leakage(self, output: str) -> Tuple[float, int]:
        """Return (fraction of system prompt shingles present, absolute count)."""
        if not self._system_shingles:
            return 0.0, 0
        out_shingles = _shingles(_words(normalize(output).canonical), self.shingle_size)
        hits = len(self._system_shingles & out_shingles)
        return hits / max(1, len(self._system_shingles)), hits

    # Credentials

    def find_credentials(self, output: str) -> List[Tuple[CredentialRule, re.Match]]:
        hits: List[Tuple[CredentialRule, re.Match]] = []
        for rule in self.credential_rules:
            for m in rule.pattern.finditer(output):
                hits.append((rule, m))
        return hits

    def find_high_entropy(self, output: str) -> List[re.Match]:
        found = []
        for m in _TOKEN_LIKE.finditer(output):
            tok = m.group(0)
            if tok.isdigit() or tok.isalpha():
                continue
            if re.fullmatch(r"[A-Fa-f0-9]{32,}", tok) and len(tok) in (32, 40, 64):
                found.append(m)
                continue
            if _shannon_entropy(tok) >= self.entropy_threshold:
                found.append(m)
        return found

    def _redact(self, output: str, spans: List[Tuple[int, int]]) -> str:
        if not spans:
            return output
        spans = sorted(spans)
        merged: List[List[int]] = []
        for s, e in spans:
            if merged and s <= merged[-1][1]:
                merged[-1][1] = max(merged[-1][1], e)
            else:
                merged.append([s, e])
        parts = []
        cursor = 0
        for s, e in merged:
            parts.append(output[cursor:s])
            parts.append(self.redact_with)
            cursor = e
        parts.append(output[cursor:])
        return "".join(parts)

    # Entry point

    def scan(self, output: str, system_prompt: Optional[str] = None) -> OutputDecision:
        if system_prompt is not None:
            self.set_system_prompt(system_prompt)
        if not isinstance(output, str):
            output = str(output)

        decision = OutputDecision(verdict=Verdict.ALLOW, stage="output", redacted=output)
        spans: List[Tuple[int, int]] = []

        # 1. Prompt leakage
        ratio, hits = self.leakage(output)
        decision.leakage_ratio = ratio
        if hits >= self.min_leak_shingles:
            if ratio >= self.leak_block_ratio:
                verdict = Verdict.BLOCK
            elif ratio >= self.leak_flag_ratio:
                verdict = Verdict.FLAG
            else:
                verdict = Verdict.ALLOW
            if verdict is not Verdict.ALLOW:
                decision.add(
                    Finding(
                        "output",
                        "prompt_leakage",
                        verdict,
                        f"{hits} system-prompt shingles reproduced ({ratio:.0%})",
                        min(1.0, ratio / self.leak_block_ratio),
                    )
                )
        if self._system_text and _LEAK_PHRASES.search(output):
            decision.add(Finding("output", "prompt_leakage_phrase", Verdict.FLAG, "output narrates its own instructions", 0.5))

        # 2. Credential shapes
        for rule, m in self.find_credentials(output):
            decision.credential_hits.append(rule.name)
            spans.append((m.start(), m.end()))
            decision.add(
                Finding(
                    "output",
                    f"credential:{rule.name}",
                    rule.verdict,
                    f"{rule.name} at offset {m.start()}",
                    1.0 if rule.verdict is Verdict.BLOCK else 0.6,
                )
            )

        # 3. High entropy tokens not already matched
        covered = [(s, e) for s, e in spans]
        for m in self.find_high_entropy(output):
            if any(s <= m.start() < e for s, e in covered):
                continue
            spans.append((m.start(), m.end()))
            decision.credential_hits.append("high_entropy")
            decision.add(
                Finding(
                    "output",
                    "credential:high_entropy",
                    Verdict.FLAG,
                    f"high-entropy token at offset {m.start()}",
                    0.5,
                )
            )

        decision.redacted = self._redact(output, spans)
        if decision.verdict is Verdict.BLOCK and self._system_shingles and ratio >= self.leak_block_ratio:
            # A leaked system prompt cannot be partially redacted meaningfully.
            decision.redacted = self.redact_with
        return decision
