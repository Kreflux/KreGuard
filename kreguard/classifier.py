"""Pluggable semantic classifier.

Patterns catch what they were written for. A classifier generalizes. KreGuard
does not ship a model; it ships an interface and a small lexicon baseline so
the pipeline works out of the box with zero dependencies.

Plug in anything that maps text to a probability of malicious intent:
a fine-tuned encoder, an embedding similarity lookup, a hosted moderation
endpoint. Wrap it in ``SafeClassifier`` and errors become fail-closed
findings instead of silent allows.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Callable, Dict, Optional, Protocol, runtime_checkable

from .normalize import NormalizedText
from .verdict import Decision, Finding, Verdict


@runtime_checkable
class Classifier(Protocol):
    """Anything that scores text for malicious intent.

    ``score`` returns a float in [0, 1]. Higher means more likely to be an
    injection or jailbreak attempt. Implementations may raise; the guard
    treats a raise as an unknown, which is never an allow.
    """

    name: str

    def score(self, text: NormalizedText) -> float: ...


@dataclass
class ClassifierResult:
    name: str
    score: Optional[float]
    error: Optional[str] = None


# Phrases carry weight; ordinary words do not. The lexicon is deliberately
# small and readable so operators can see exactly why a score moved.
_DEFAULT_LEXICON: Dict[str, float] = {
    "ignore": 0.6,
    "disregard": 0.7,
    "override": 0.7,
    "bypass": 0.8,
    "jailbreak": 1.5,
    "unrestricted": 0.9,
    "uncensored": 1.0,
    "unfiltered": 0.9,
    "system prompt": 1.2,
    "hidden instructions": 1.2,
    "reveal": 0.5,
    "pretend": 0.6,
    "roleplay": 0.4,
    "developer mode": 1.3,
    "no rules": 1.0,
    "no restrictions": 1.1,
    "without restrictions": 1.1,
    "you must": 0.5,
    "you are now": 0.9,
    "from now on": 0.7,
    "do anything": 0.8,
    "exfiltrate": 1.5,
    "secret": 0.4,
    "password": 0.5,
    "api key": 0.7,
    "credentials": 0.6,
    "do not tell": 0.9,
    "don't tell": 0.9,
    "confidential": 0.4,
    "base64": 0.6,
    "rot13": 0.8,
    "sudo": 0.6,
    "rm -rf": 1.2,
    "execute": 0.4,
    "delete": 0.3,
    "transfer": 0.3,
    "admin": 0.4,
    "root": 0.3,
    "instructions": 0.4,
    "prompt": 0.4,
    "evil": 0.6,
    "fictional": 0.4,
    "hypothetically": 0.5,
    "previous": 0.3,
    "above": 0.2,
}

_BENIGN_LEXICON: Dict[str, float] = {
    "please help": 0.4,
    "thank you": 0.4,
    "recipe": 0.3,
    "weather": 0.3,
    "schedule": 0.3,
    "summarize": 0.3,
    "translate": 0.2,
    "explain": 0.2,
}


class LexiconClassifier:
    """Baseline classifier: weighted phrase hits through a logistic squash.

    It is intentionally modest. Its job is to prove the interface and to add
    a little recall on paraphrases the regex rules miss. Replace it with a
    real model in production.
    """

    name = "lexicon"

    def __init__(
        self,
        lexicon: Optional[Dict[str, float]] = None,
        benign: Optional[Dict[str, float]] = None,
        bias: float = -2.2,
        slope: float = 1.0,
    ) -> None:
        self.lexicon = dict(_DEFAULT_LEXICON if lexicon is None else lexicon)
        self.benign = dict(_BENIGN_LEXICON if benign is None else benign)
        self.bias = bias
        self.slope = slope
        self._compiled = {
            phrase: re.compile(r"\b" + re.escape(phrase) + r"\b", re.IGNORECASE)
            for phrase in list(self.lexicon) + list(self.benign)
        }

    def _energy(self, view: str) -> float:
        energy = 0.0
        for phrase, weight in self.lexicon.items():
            hits = len(self._compiled[phrase].findall(view))
            if hits:
                energy += weight * min(hits, 3) ** 0.5
        for phrase, weight in self.benign.items():
            if self._compiled[phrase].search(view):
                energy -= weight
        # Very long inputs dilute isolated hits; very short ones concentrate them.
        words = max(1, len(view.split()))
        energy *= 1.0 / (1.0 + math.log10(max(1.0, words / 40.0)))
        return energy

    def score(self, text: NormalizedText) -> float:
        best = 0.0
        for view in text.views:
            energy = self._energy(view)
            p = 1.0 / (1.0 + math.exp(-(self.slope * energy + self.bias)))
            best = max(best, p)
        return best


class CallableClassifier:
    """Adapt a plain ``Callable[[str], float]`` to the Classifier protocol."""

    def __init__(self, fn: Callable[[str], float], name: str = "callable") -> None:
        self._fn = fn
        self.name = name

    def score(self, text: NormalizedText) -> float:
        return float(self._fn(text.canonical))


class SafeClassifier:
    """Wrap a classifier so failures are recorded, never swallowed."""

    def __init__(self, inner: Classifier) -> None:
        self.inner = inner
        self.name = getattr(inner, "name", type(inner).__name__)

    def run(self, text: NormalizedText) -> ClassifierResult:
        try:
            value = self.inner.score(text)
        except Exception as exc:  # noqa: BLE001 - any failure must be recorded
            return ClassifierResult(self.name, None, f"{type(exc).__name__}: {exc}")
        try:
            value = float(value)
        except (TypeError, ValueError):
            return ClassifierResult(self.name, None, f"non-numeric score {value!r}")
        if math.isnan(value) or math.isinf(value):
            return ClassifierResult(self.name, None, f"invalid score {value!r}")
        return ClassifierResult(self.name, min(1.0, max(0.0, value)))

    def decide(
        self,
        text: NormalizedText,
        flag_threshold: float,
        block_threshold: float,
        on_error: Verdict,
    ) -> Decision:
        result = self.run(text)
        decision = Decision(verdict=Verdict.ALLOW, stage="classifier")
        if result.score is None:
            verdict = on_error if on_error is not Verdict.ALLOW else Verdict.BLOCK
            decision.add(
                Finding(
                    source="classifier",
                    rule=f"{result.name}:error",
                    verdict=verdict,
                    detail=result.error or "unknown error",
                    # An error is unknown, not evidence. Score stays zero so the
                    # verdict is carried by on_error alone.
                    score=0.0,
                )
            )
            return decision
        decision.score = result.score
        if result.score >= block_threshold:
            verdict = Verdict.BLOCK
        elif result.score >= flag_threshold:
            verdict = Verdict.FLAG
        else:
            verdict = Verdict.ALLOW
        decision.verdict = verdict
        if verdict is not Verdict.ALLOW:
            decision.findings.append(
                Finding(
                    source="classifier",
                    rule=result.name,
                    verdict=verdict,
                    detail=f"p(malicious)={result.score:.3f}",
                    score=result.score,
                )
            )
        return decision
