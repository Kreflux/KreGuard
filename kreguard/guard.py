"""The Guard: one object that wires every stage together.

Input path:   normalize -> patterns -> classifier -> judge (ambiguous only)
Output path:  prompt leakage -> credential shapes -> redaction
Enforcement:  tool allowlist, egress allowlist

The two enforcement gates do not consult the text verdicts. They are the
floor beneath everything else.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Optional
from urllib.parse import urlsplit

from .audit import AuditLog
from .classifier import Classifier, SafeClassifier
from .judge import Judge, apply_judge
from .normalize import normalize
from .output_scan import OutputDecision, OutputScanner
from .patterns import PatternScanner
from .permissions import EgressPolicy, ToolPolicy
from .verdict import Decision, Finding, Verdict, fail_closed


@dataclass
class GuardConfig:
    flag_threshold: float = 0.35
    block_threshold: float = 0.8
    max_input_chars: int = 32_000
    on_error: Verdict = Verdict.BLOCK
    judge_can_clear: bool = True
    judge_context: Optional[str] = None
    empty_input_verdict: Verdict = Verdict.ALLOW
    # With a learning classifier, teach it from hard evidence (a pattern rule
    # that blocks outright, or a judge that escalates). Only ever toward
    # blocking; relaxing takes explicit feedback from a person.
    auto_learn: bool = True

    def __post_init__(self) -> None:
        if not (0.0 <= self.flag_threshold < self.block_threshold <= 1.0):
            raise ValueError("thresholds must satisfy 0 <= flag < block <= 1")
        if self.on_error is Verdict.ALLOW:
            raise ValueError("on_error cannot be ALLOW: KreGuard never fails open")


class Guard:
    def __init__(
        self,
        config: Optional[GuardConfig] = None,
        patterns: Optional[PatternScanner] = None,
        classifier: Optional[Classifier] = None,
        judge: Optional[Judge] = None,
        output_scanner: Optional[OutputScanner] = None,
        tool_policy: Optional[ToolPolicy] = None,
        egress_policy: Optional[EgressPolicy] = None,
        system_prompt: Optional[str] = None,
        audit: Optional[AuditLog] = None,
    ) -> None:
        self.config = config or GuardConfig()
        self.patterns = patterns or PatternScanner(
            flag_threshold=self.config.flag_threshold,
            block_threshold=self.config.block_threshold,
        )
        self.classifier = SafeClassifier(classifier) if classifier is not None else None
        self._learner = classifier if classifier is not None and callable(getattr(classifier, "learn", None)) else None
        self.judge = judge
        self.output_scanner = output_scanner or OutputScanner(system_prompt=system_prompt)
        if system_prompt and output_scanner is not None:
            self.output_scanner.set_system_prompt(system_prompt)
        # Empty policies deny everything. That is the point.
        self.tool_policy = tool_policy or ToolPolicy()
        self.egress_policy = egress_policy or EgressPolicy()
        self.audit = audit

    # Input

    def check_input(self, text: str) -> Decision:
        try:
            decision = self._check_input(text)
        except Exception as exc:  # noqa: BLE001 - never fail open
            decision = fail_closed("input", exc, self.config.on_error)
        self._record("input", decision, text if isinstance(text, str) else None)
        return decision

    def _check_input(self, text: str) -> Decision:
        if text is None or (isinstance(text, str) and not text.strip()):
            return Decision(verdict=self.config.empty_input_verdict, stage="input")
        if not isinstance(text, str):
            text = str(text)
        if len(text) > self.config.max_input_chars:
            d = Decision(verdict=Verdict.BLOCK, stage="input", score=1.0)
            d.findings.append(Finding("input", "too_long", Verdict.BLOCK, f"{len(text)} > {self.config.max_input_chars} chars", 1.0))
            return d

        normalized = normalize(text)
        decision = self.patterns.scan(normalized)
        pattern_score = decision.score
        hard_evidence = decision.verdict is Verdict.BLOCK

        if self.classifier is not None:
            cls = self.classifier.decide(
                normalized,
                self.config.flag_threshold,
                self.config.block_threshold,
                self.config.on_error,
            )
            decision = decision.merge(cls)
            # Blend with noisy-or so two mid-strength signals reinforce,
            # but damp the classifier so it cannot block alone on a whisper.
            combined = 1.0 - (1.0 - pattern_score) * (1.0 - 0.85 * cls.score)
            decision.score = min(1.0, max(pattern_score, cls.score, combined))
            if decision.verdict is not Verdict.BLOCK:
                if decision.score >= self.config.block_threshold:
                    decision.verdict = Verdict.BLOCK
                elif decision.score >= self.config.flag_threshold:
                    decision.verdict = Verdict.FLAG

        if decision.verdict is Verdict.FLAG and self.judge is not None:
            decision = apply_judge(
                self.judge,
                normalized.canonical,
                self.config.judge_context,
                decision,
                can_clear=self.config.judge_can_clear,
                on_error=self.config.on_error,
            )

        if not hard_evidence:
            hard_evidence = any(
                f.source == "judge" and f.verdict is Verdict.BLOCK and not f.rule.endswith(":error")
                for f in decision.findings
            )
        if hard_evidence and self.config.auto_learn and self._learner is not None:
            try:
                self._learner.learn(normalized, True, source="auto")
            except Exception:  # noqa: BLE001 - learning must never change a verdict
                pass

        decision.stage = decision.stage or "input"
        return decision

    # Output

    def check_output(self, text: str, system_prompt: Optional[str] = None) -> OutputDecision:
        try:
            decision = self.output_scanner.scan(text, system_prompt)
        except Exception as exc:  # noqa: BLE001
            base = fail_closed("output", exc, self.config.on_error)
            decision = OutputDecision(
                verdict=base.verdict,
                findings=base.findings,
                score=base.score,
                stage="output",
                redacted=self.output_scanner.redact_with,
            )
        self._record("output", decision, text if isinstance(text, str) else None)
        return decision

    # Enforcement

    def authorize_tool(self, tool: str, args: Optional[Mapping[str, Any]] = None) -> Decision:
        try:
            decision = self.tool_policy.authorize(tool, args)
        except Exception as exc:  # noqa: BLE001
            decision = fail_closed("tools", exc, Verdict.BLOCK)
        # Argument values stay out of the log; names are enough to investigate.
        keys = sorted(str(k) for k in args) if isinstance(args, Mapping) else None
        self._record(
            "tool",
            decision,
            display=tool if isinstance(tool, str) else repr(tool),
            extra={"arg_names": keys} if keys else None,
        )
        return decision

    def authorize_egress(self, url: str) -> Decision:
        try:
            decision = self.egress_policy.authorize(url)
        except Exception as exc:  # noqa: BLE001
            decision = fail_closed("egress", exc, Verdict.BLOCK)
        self._record("egress", decision, url if isinstance(url, str) else None, display=_safe_url(url))
        return decision

    # Learning

    @property
    def learner(self):
        """The text classifier if it learns, else None."""
        return self._learner

    def feedback_input(self, text: str, attack: bool) -> dict:
        """Tell the guard what an input really was. This is how it adapts.

        ``attack=True`` makes the guard block this text and its close
        paraphrases from now on. ``attack=False`` corrects a false positive:
        the text stops matching remembered attacks and the model is pushed
        toward treating it as safe. Pattern rules are static and are not
        affected; fix those by editing the rules.
        """
        if self._learner is None:
            raise AdaptationError("the classifier in use does not learn; use AdaptiveClassifier")
        if not isinstance(text, str) or not text.strip():
            raise ValueError("text must be a non-empty string")
        result = self._learner.learn(text, bool(attack), source="feedback")
        self._event("feedback", {"target": "input", "attack": bool(attack), **result.as_dict()}, text)
        return result.as_dict()

    def feedback_egress(self, url: str, malicious: bool) -> dict:
        """Tell the guard that a destination is malicious (blocked from now on,
        with its subdomains, and lookalikes score high) or safe (forgiven, and
        the request shape counts as normal)."""
        learner = self.egress_policy.learner
        if learner is None:
            raise AdaptationError("this egress policy has no learner; give it AdaptiveEgress()")
        result = learner.report(url, bool(malicious))
        self._event("feedback", {"target": "egress", "malicious": bool(malicious), "host": result.get("host")})
        return result

    def save_models(self) -> list:
        """Persist every learner that has somewhere to save. Returns what was saved."""
        saved = []
        for part in (self._learner, self.egress_policy.learner):
            if part is not None and getattr(part, "path", None) is not None:
                part.save()
                saved.append(str(part.path))
        return saved

    def model_stats(self) -> dict:
        out: dict = {}
        if self._learner is not None and hasattr(self._learner, "stats"):
            out["text"] = self._learner.stats()
        if self.egress_policy.learner is not None:
            out["egress"] = self.egress_policy.learner.stats()
        return out

    def _event(self, kind: str, fields: Mapping[str, Any], subject: Optional[str] = None) -> None:
        if self.audit is None:
            return
        try:
            self.audit.event(kind, fields, subject)
        except Exception:  # noqa: BLE001
            pass

    def _record(self, kind: str, decision: Decision, subject: Optional[str] = None, display: Optional[str] = None, extra: Optional[Mapping[str, Any]] = None) -> None:
        if self.audit is None:
            return
        try:
            self.audit.record(kind, decision, subject, display, extra)
        except Exception:  # noqa: BLE001 - auditing never changes a verdict
            pass


class AdaptationError(RuntimeError):
    """Feedback was sent to a guard that has nothing to teach."""


def _safe_url(url: Any) -> Optional[str]:
    """scheme://host/path only. Query strings and userinfo often hold secrets."""
    if not isinstance(url, str):
        return None
    try:
        parts = urlsplit(url.strip())
        host = parts.hostname or ""
    except ValueError:
        return "<unparseable>"
    return f"{parts.scheme}://{host}{parts.path}"[:200]
