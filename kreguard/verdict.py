"""Verdict model shared by every KreGuard check.

Three verdicts, strictly ordered: allow < flag < block. Combining verdicts
always takes the most severe one, so no stage can quietly downgrade another.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Iterable, List, Optional


class Verdict(str, Enum):
    ALLOW = "allow"
    FLAG = "flag"
    BLOCK = "block"

    @property
    def rank(self) -> int:
        return _RANK[self]

    def __lt__(self, other: object) -> bool:  # type: ignore[override]
        if not isinstance(other, Verdict):
            return NotImplemented
        return self.rank < other.rank

    def __le__(self, other: object) -> bool:  # type: ignore[override]
        if not isinstance(other, Verdict):
            return NotImplemented
        return self.rank <= other.rank

    def __gt__(self, other: object) -> bool:  # type: ignore[override]
        if not isinstance(other, Verdict):
            return NotImplemented
        return self.rank > other.rank

    def __ge__(self, other: object) -> bool:  # type: ignore[override]
        if not isinstance(other, Verdict):
            return NotImplemented
        return self.rank >= other.rank

    def __hash__(self) -> int:
        return hash(self.value)


_RANK = {Verdict.ALLOW: 0, Verdict.FLAG: 1, Verdict.BLOCK: 2}


def worst(verdicts: Iterable[Verdict]) -> Verdict:
    """Return the most severe verdict. Empty input is ALLOW."""
    result = Verdict.ALLOW
    for v in verdicts:
        if v > result:
            result = v
    return result


@dataclass(frozen=True)
class Finding:
    """A single piece of evidence produced by one stage."""

    source: str
    rule: str
    verdict: Verdict
    detail: str = ""
    score: float = 0.0

    def as_dict(self) -> dict:
        return {
            "source": self.source,
            "rule": self.rule,
            "verdict": self.verdict.value,
            "detail": self.detail,
            "score": round(self.score, 4),
        }


@dataclass
class Decision:
    """The outcome of a check, with the evidence that produced it."""

    verdict: Verdict
    findings: List[Finding] = field(default_factory=list)
    score: float = 0.0
    stage: Optional[str] = None

    @property
    def allowed(self) -> bool:
        return self.verdict is Verdict.ALLOW

    @property
    def blocked(self) -> bool:
        return self.verdict is Verdict.BLOCK

    def add(self, finding: Finding) -> None:
        self.findings.append(finding)
        if finding.verdict > self.verdict:
            self.verdict = finding.verdict
        if finding.score > self.score:
            self.score = finding.score

    def merge(self, other: "Decision") -> "Decision":
        merged = Decision(
            verdict=worst([self.verdict, other.verdict]),
            findings=[*self.findings, *other.findings],
            score=max(self.score, other.score),
            stage=other.stage or self.stage,
        )
        return merged

    def as_dict(self) -> dict:
        return {
            "verdict": self.verdict.value,
            "score": round(self.score, 4),
            "stage": self.stage,
            "findings": [f.as_dict() for f in self.findings],
        }


def fail_closed(source: str, error: BaseException, verdict: Verdict = Verdict.BLOCK) -> Decision:
    """Build the decision returned when a stage raises.

    The default is BLOCK. Callers may downgrade to FLAG through configuration,
    but there is deliberately no way to produce ALLOW from an error.
    """
    if verdict is Verdict.ALLOW:
        verdict = Verdict.BLOCK
    return Decision(
        verdict=verdict,
        findings=[
            Finding(
                source=source,
                rule="internal_error",
                verdict=verdict,
                detail=f"{type(error).__name__}: {error}",
                score=1.0,
            )
        ],
        score=1.0,
        stage=source,
    )
