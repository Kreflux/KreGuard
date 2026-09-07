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
    ) -> None:
        self.config = config or GuardConfig()
        self.patterns = patterns or PatternScanner(
            flag_threshold=self.config.flag_threshold,
            block_threshold=self.config.block_threshold,
        )
        self.classifier = SafeClassifier(classifier) if classifier is not None else None
        self.judge = judge
        self.output_scanner = output_scanner or OutputScanner(system_prompt=system_prompt)
        if system_prompt and output_scanner is not None:
            self.output_scanner.set_system_prompt(system_prompt)
        # Empty policies deny everything. That is the point.
        self.tool_policy = tool_policy or ToolPolicy()
        self.egress_policy = egress_policy or EgressPolicy()

    # Input

    def check_input(self, text: str) -> Decision:
        try:
            return self._check_input(text)
        except Exception as exc:  # noqa: BLE001 - never fail open
            return fail_closed("input", exc, self.config.on_error)

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

        decision.stage = decision.stage or "input"
        return decision

    # Output

    def check_output(self, text: str, system_prompt: Optional[str] = None) -> OutputDecision:
        try:
            return self.output_scanner.scan(text, system_prompt)
        except Exception as exc:  # noqa: BLE001
            base = fail_closed("output", exc, self.config.on_error)
            return OutputDecision(
                verdict=base.verdict,
                findings=base.findings,
                score=base.score,
                stage="output",
                redacted=self.output_scanner.redact_with,
            )

    # Enforcement

    def authorize_tool(self, tool: str, args: Optional[Mapping[str, Any]] = None) -> Decision:
        try:
            return self.tool_policy.authorize(tool, args)
        except Exception as exc:  # noqa: BLE001
            return fail_closed("tools", exc, Verdict.BLOCK)

    def authorize_egress(self, url: str) -> Decision:
        try:
            return self.egress_policy.authorize(url)
        except Exception as exc:  # noqa: BLE001
            return fail_closed("egress", exc, Verdict.BLOCK)
