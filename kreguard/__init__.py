"""
KreGuard: by Kreflux
An independent lab building Kreflux.
Kre, to create. Flux, to change.
https://kreflux.com

KreGuard is a guardrail layer that sits between an LLM application and the
world. It assumes the model will eventually be tricked. Text defenses
(normalization, pattern matching, semantic classification, an LLM judge)
are advisory: they raise the cost of an attack. Permission gates on tools
and egress are enforcement: they bound the damage when an attack succeeds.

Every check returns one of three verdicts: allow, flag, block. Every check
fails closed. An internal error is never an allow.

The core has no runtime dependencies outside the Python standard library.
"""

from .classifier import Classifier, LexiconClassifier
from .guard import Guard, GuardConfig
from .judge import Judge, JudgeResult, PromptJudge
from .normalize import NormalizedText, normalize
from .output_scan import OutputDecision, OutputScanner
from .patterns import PatternScanner, Rule
from .permissions import EgressPolicy, ToolPolicy
from .verdict import Decision, Finding, Verdict, worst

__all__ = [
    "Classifier",
    "Decision",
    "EgressPolicy",
    "Finding",
    "Guard",
    "GuardConfig",
    "Judge",
    "JudgeResult",
    "LexiconClassifier",
    "NormalizedText",
    "OutputDecision",
    "OutputScanner",
    "PatternScanner",
    "PromptJudge",
    "Rule",
    "ToolPolicy",
    "Verdict",
    "normalize",
    "worst",
]
