"""Adaptive defenses: components that keep learning while they guard."""

from .egress import AdaptiveEgress
from .model import ModelError, OnlineLogReg
from .text import AdaptiveClassifier, LearnResult

__all__ = ["AdaptiveClassifier", "AdaptiveEgress", "LearnResult", "ModelError", "OnlineLogReg"]
