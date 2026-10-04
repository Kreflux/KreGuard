"""The adaptive text classifier: a model that keeps learning.

``AdaptiveClassifier`` implements the same ``Classifier`` interface as every
other KreGuard classifier, so it drops into a Guard unchanged. What differs is
that it is not frozen. It learns in three ways, at three speeds.

1. Weights (generalization). A sparse logistic regression over hashed word
   and character n-grams, updated online. It starts from a seed corpus so it
   is useful on day one, then keeps moving with traffic. A confirmed example
   is applied until the model fits it, so one correction shifts the paraphrases
   of that example too, not only the exact string.

2. Memory (one-shot). Every confirmed attack is remembered as a set of hashed
   phrase shingles. A near-duplicate of a confirmed attack scores high at once,
   without waiting for the weights to catch up. Telling the model that a
   remembered text is safe evicts it, which is how false positives are fixed.

3. Self-training from hard evidence. When a deterministic stage is certain
   (a pattern rule blocks outright, a judge escalates), the guard teaches the
   classifier. This is how it picks up the paraphrases the rules miss.

The safety properties matter more than the learning:

* Learning is advisory. Like every text defense here it can only raise or
  lower the classifier's own score. It never overrides a pattern block and it
  never touches permissions, which stay static policy.
* Self-training only ever moves toward blocking. Nothing is auto-learned as
  benign: a judge or a clever input could otherwise teach the model to wave
  attacks through. Relaxing takes a human saying so.
* It is rate limited, de-duplicated, and margin based (it skips what it
  already knows), so a flood of crafted input cannot move it quickly.
* Weights are clipped and the model file is checksummed and validated.
"""

from __future__ import annotations

import copy
import hashlib
import shutil
import threading
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Deque, Dict, FrozenSet, List, Mapping, Optional, Tuple, Union

from ..normalize import NormalizedText, normalize
from .features import hash_features, is_uninformative, shingles, text_features, top_contributions
from .model import ModelError, OnlineLogReg, load_json, save_json
from .seed import ATTACKS, BENIGN

_KIND = "adaptive-text"
_SEEDED: Optional[OnlineLogReg] = None  # the trained seed model, built once per process
_FIT_MARGIN = 0.1       # stop fitting an example once the model is this close
_UNSURE = 0.15          # skip the update if the model already agrees this much
_MAX_FIT_STEPS = 8


@dataclass
class LearnResult:
    updated: bool = False
    p_before: float = 0.0
    p_after: float = 0.0
    memorized: bool = False
    evicted: int = 0
    skipped: Optional[str] = None

    def as_dict(self) -> Dict[str, Any]:
        return {
            "updated": self.updated,
            "p_before": round(self.p_before, 4),
            "p_after": round(self.p_after, 4),
            "memorized": self.memorized,
            "evicted": self.evicted,
            "skipped": self.skipped,
        }


class AdaptiveClassifier:
    name = "adaptive"

    def __init__(
        self,
        *,
        path: Union[str, Path, None] = None,
        seed: bool = True,
        autosave_every: int = 0,
        memory_size: int = 4096,
        memory_threshold: float = 0.6,
        auto_per_hour: int = 500,
        backup_interval: float = 6 * 3600.0,
        clock: Callable[[], float] = time.time,
        model: Optional[OnlineLogReg] = None,
    ) -> None:
        if not 0.2 <= memory_threshold <= 1.0:
            raise ValueError("memory_threshold must be between 0.2 and 1.0")
        self.path = Path(path) if path is not None else None
        self.autosave_every = autosave_every
        self.memory_size = memory_size
        self.memory_threshold = memory_threshold
        self.auto_per_hour = auto_per_hour
        self.backup_interval = backup_interval
        self._clock = clock
        self._lock = threading.RLock()
        self.model = model or OnlineLogReg(lr=0.5)
        # Memory of confirmed attacks, with an inverted index for fast lookup.
        self._mem: Dict[int, FrozenSet[int]] = {}
        self._mem_src: Dict[int, str] = {}
        self._index: Dict[int, set] = {}
        self._next_id = 0
        self._auto_seen: Deque[int] = deque(maxlen=8192)
        self._auto_seen_set: set = set()
        self._auto_times: Deque[float] = deque()
        self.counts: Dict[str, int] = {
            "feedback_attack": 0, "feedback_safe": 0, "auto_attack": 0,
            "skipped": 0, "tp": 0, "fp": 0, "fn": 0, "tn": 0,
        }
        self.recent_accuracy: Optional[float] = None
        self.seeded = False
        self.last_save_error: Optional[str] = None
        self._since_save = 0

        if self.path is not None and self.path.exists():
            self._load(self.path)
        elif seed and model is None:
            self._seed()

    # Scoring

    def score(self, text: NormalizedText) -> float:
        feats = hash_features(text_features(text), self.model.dim)
        p = self.model.predict(feats)
        mem, _ = self._memory_score(shingles(text))
        return max(p, mem)

    def explain(self, text: Union[str, NormalizedText], limit: int = 5) -> Dict[str, Any]:
        nt = text if isinstance(text, NormalizedText) else normalize(str(text))
        named = text_features(nt)
        feats = hash_features(named, self.model.dim)
        mem, jac = self._memory_score(shingles(nt))
        return {
            "model": round(self.model.predict(feats), 4),
            "memory": round(mem, 4),
            "similarity": round(jac, 3),
            "top": [(n, round(c, 3)) for n, c in top_contributions(named, self.model.dim, self.model.w, limit, skip=is_uninformative)],
        }

    def describe(self, text: NormalizedText) -> str:
        """One line for the decision's evidence: why the score is what it is."""
        info = self.explain(text, limit=4)
        parts = []
        if info["memory"] > 0:
            parts.append(f"near-duplicate of a confirmed attack (similarity {info['similarity']})")
        pushes = [f"{n.split(':', 1)[-1].replace('_', ' ')} ({c:+.1f})" for n, c in info["top"] if c > 0.15]
        if pushes:
            parts.append("pushed by " + ", ".join(pushes[:3]))
        return "; ".join(parts)

    # Memory

    def _memory_score(self, sh: List[int]) -> Tuple[float, float]:
        if not sh or not self._mem:
            return 0.0, 0.0
        # Held under the lock: a concurrent learn() edits these sets, and
        # iterating one mid-edit would raise and turn into a false block.
        with self._lock:
            counts: Dict[int, int] = {}
            index = self._index
            for s in sh:
                for mid in index.get(s, ()):
                    counts[mid] = counts.get(mid, 0) + 1
            best = 0.0
            nq = len(sh)
            for mid, c in counts.items():
                j = c / (nq + len(self._mem[mid]) - c)
                if j > best:
                    best = j
        if best >= self.memory_threshold:
            return 0.85 + 0.15 * best, best
        return 0.0, best

    def _remember(self, sh: List[int], src: str) -> bool:
        if not sh:
            return False
        fs = frozenset(sh)
        _, best = self._memory_score(sh)
        if best >= 0.95:
            return False  # already known
        mid = self._next_id
        self._next_id += 1
        self._mem[mid] = fs
        self._mem_src[mid] = src
        for s in fs:
            self._index.setdefault(s, set()).add(mid)
        while len(self._mem) > self.memory_size:
            self._evict(min(self._mem))
        return True

    def _evict(self, mid: int) -> None:
        for s in self._mem.pop(mid, ()):
            bucket = self._index.get(s)
            if bucket is not None:
                bucket.discard(mid)
                if not bucket:
                    del self._index[s]
        self._mem_src.pop(mid, None)

    def _evict_similar(self, sh: List[int]) -> int:
        if not sh or not self._mem:
            return 0
        counts: Dict[int, int] = {}
        for s in sh:
            for mid in self._index.get(s, ()):
                counts[mid] = counts.get(mid, 0) + 1
        doomed = [mid for mid, c in counts.items() if c / (len(sh) + len(self._mem[mid]) - c) >= self.memory_threshold]
        for mid in doomed:
            self._evict(mid)
        return len(doomed)

    # Learning

    def learn(self, text: Union[str, NormalizedText], attack: bool, *, source: str = "feedback") -> LearnResult:
        """Teach the model one labeled example.

        ``source="feedback"`` is a human (or trusted system) giving ground
        truth. ``source="auto"`` is the guard teaching itself from hard
        evidence; it can only mark attacks, is rate limited and de-duplicated.
        """
        if source not in ("feedback", "auto"):
            raise ValueError("source must be 'feedback' or 'auto'")
        nt = text if isinstance(text, NormalizedText) else normalize(str(text))
        result = LearnResult()
        with self._lock:
            sh = shingles(nt)
            if source == "auto":
                reason = self._auto_gate(nt, attack)
                if reason:
                    result.skipped = reason
                    self.counts["skipped"] += 1
                    return result
            feats = hash_features(text_features(nt), self.model.dim)
            p_model = self.model.predict(feats)
            mem, _ = self._memory_score(sh)
            result.p_before = max(p_model, mem)
            target = 1 if attack else 0

            if source == "feedback":
                self._account(attack, result.p_before)
                self.counts["feedback_attack" if attack else "feedback_safe"] += 1
            else:
                self.counts["auto_attack"] += 1

            # Fit until the model agrees, but only if it does not already.
            steps = 0
            max_steps = _MAX_FIT_STEPS if source == "feedback" else 2
            p = p_model
            while abs(p - target) > (_FIT_MARGIN if steps else _UNSURE) and steps < max_steps:
                self.model.update(feats, target)
                p = self.model.predict(feats)
                steps += 1
            result.updated = steps > 0

            if attack:
                result.memorized = self._remember(sh, source)
            else:
                result.evicted = self._evict_similar(sh)
            mem_after, _ = self._memory_score(sh)
            result.p_after = max(p, mem_after)

            self._since_save += 1
            if self.path is not None and self.autosave_every and self._since_save >= self.autosave_every:
                self._autosave()
        return result

    def _auto_gate(self, nt: NormalizedText, attack: bool) -> Optional[str]:
        if not attack:
            return "auto_learning_never_marks_benign"
        now = self._clock()
        while self._auto_times and now - self._auto_times[0] > 3600.0:
            self._auto_times.popleft()
        if len(self._auto_times) >= self.auto_per_hour:
            return "rate_limited"
        key = int.from_bytes(hashlib.blake2b((nt.folded or nt.canonical).encode("utf-8", "replace"), digest_size=8).digest(), "big")
        if key in self._auto_seen_set:
            return "duplicate"
        if len(self._auto_seen) == self._auto_seen.maxlen:
            self._auto_seen_set.discard(self._auto_seen[0])
        self._auto_seen.append(key)
        self._auto_seen_set.add(key)
        self._auto_times.append(now)
        return None

    def _account(self, attack: bool, p_before: float) -> None:
        """Test-then-train: grade the prediction made *before* learning."""
        predicted = p_before >= 0.5
        key = ("tp" if attack else "fp") if predicted else ("fn" if attack else "tn")
        self.counts[key] += 1
        correct = 1.0 if predicted == attack else 0.0
        self.recent_accuracy = correct if self.recent_accuracy is None else 0.9 * self.recent_accuracy + 0.1 * correct

    def train(self, examples: List[Tuple[str, bool]], epochs: int = 1) -> int:
        """Bulk feedback, for example a labeled export of past traffic."""
        n = 0
        for _ in range(max(1, epochs)):
            for text, attack in examples:
                self.learn(text, attack)
                n += 1
        return n

    def _seed(self) -> None:
        """Deterministic cold start from the built-in corpus.

        Training is identical every time, so it is done once per process and
        copied. That keeps a second Guard in the same process instant.
        """
        global _SEEDED
        if _SEEDED is None:
            _SEEDED = self._train_seed()
        self.model = copy.deepcopy(_SEEDED)
        self.seeded = True

    def _train_seed(self) -> OnlineLogReg:
        model = OnlineLogReg(lr=0.5)
        pos = [hash_features(text_features(normalize(t)), model.dim) for t in ATTACKS]
        neg = [hash_features(text_features(normalize(t)), model.dim) for t in BENIGN]
        order: List[Tuple[Dict[int, float], int]] = []
        n = max(len(pos), len(neg))
        for i in range(n):
            order.append((pos[i % len(pos)], 1))
            order.append((neg[(i * len(neg)) // n], 0))
        for _ in range(8):
            for feats, y in order:
                model.update(feats, y)
        return model

    def reset(self) -> None:
        """Forget everything learned and return to the built-in seed."""
        with self._lock:
            self._mem.clear()
            self._mem_src.clear()
            self._index.clear()
            self._auto_seen.clear()
            self._auto_seen_set.clear()
            self._auto_times.clear()
            for k in self.counts:
                self.counts[k] = 0
            self.recent_accuracy = None
            self._seed()

    # Introspection

    def stats(self) -> Dict[str, Any]:
        with self._lock:
            c = self.counts
            graded = c["tp"] + c["fp"] + c["fn"] + c["tn"]
            return {
                "name": self.name,
                "updates": self.model.updates,
                "weights": len(self.model.w),
                "memory": len(self._mem),
                "memory_auto": sum(1 for s in self._mem_src.values() if s == "auto"),
                "feedback_attack": c["feedback_attack"],
                "feedback_safe": c["feedback_safe"],
                "auto_attack": c["auto_attack"],
                "skipped": c["skipped"],
                "graded_before_learning": graded,
                "accuracy_before_learning": round((c["tp"] + c["tn"]) / graded, 4) if graded else None,
                "recent_accuracy": round(self.recent_accuracy, 4) if self.recent_accuracy is not None else None,
                "confusion": {k: c[k] for k in ("tp", "fp", "fn", "tn")},
                "persistent": self.path is not None,
                "last_save_error": self.last_save_error,
            }

    # Persistence

    def save(self, path: Union[str, Path, None] = None) -> None:
        target = Path(path) if path is not None else self.path
        if target is None:
            raise ValueError("no path to save to")
        with self._lock:
            payload = {
                "model": self.model.to_dict(),
                "memory": [[mid, self._mem_src.get(mid, "feedback"), sorted(fs)] for mid, fs in sorted(self._mem.items())],
                "next_id": self._next_id,
                "counts": self.counts,
                "recent_accuracy": self.recent_accuracy,
                "auto_seen": list(self._auto_seen),
                "memory_threshold": self.memory_threshold,
            }
            # Keep a recent known-good copy to roll back to if the model is
            # ever poisoned. Refreshed at most once per interval, judged by the
            # backup file itself so it holds across separate processes.
            bak = Path(str(target) + ".bak")
            try:
                due = target.exists() and (not bak.exists() or time.time() - bak.stat().st_mtime >= self.backup_interval)
                if due:
                    shutil.copy2(target, bak)
            except OSError:
                pass
            save_json(target, _KIND, payload)
            self._since_save = 0

    def _autosave(self) -> None:
        try:
            self.save()
            self.last_save_error = None
        except (OSError, ValueError) as exc:
            self.last_save_error = f"{type(exc).__name__}: {exc}"

    def _load(self, path: Path) -> None:
        payload = load_json(path, _KIND)
        try:
            self.model = OnlineLogReg.from_dict(payload["model"])
            mem = payload["memory"]
            if not isinstance(mem, list) or len(mem) > self.memory_size * 2:
                raise ModelError("memory is malformed")
            self._mem.clear(); self._mem_src.clear(); self._index.clear()
            for row in mem:
                if not (isinstance(row, list) and len(row) == 3):
                    raise ModelError("memory entry is malformed")
                mid, src, sh = row
                if isinstance(mid, bool) or not isinstance(mid, int) or mid < 0 or src not in ("feedback", "auto") or not isinstance(sh, list) or len(sh) > 20000:
                    raise ModelError("memory entry is malformed")
                for s in sh:
                    if isinstance(s, bool) or not isinstance(s, int) or not 0 <= s <= 0xFFFFFFFF:
                        raise ModelError("memory shingle out of range")
                fs = frozenset(sh)
                self._mem[mid] = fs
                self._mem_src[mid] = src
                for s in fs:
                    self._index.setdefault(s, set()).add(mid)
            nid = payload["next_id"]
            if isinstance(nid, bool) or not isinstance(nid, int) or nid < 0:
                raise ModelError("next_id out of range")
            self._next_id = max(nid, (max(self._mem) + 1) if self._mem else 0)
            counts = payload["counts"]
            if not isinstance(counts, dict):
                raise ModelError("counts are malformed")
            for k in self.counts:
                v = counts.get(k, 0)
                if isinstance(v, bool) or not isinstance(v, int) or v < 0:
                    raise ModelError("count out of range")
                self.counts[k] = v
            ra = payload.get("recent_accuracy")
            if ra is not None and (isinstance(ra, bool) or not isinstance(ra, (int, float)) or not 0.0 <= ra <= 1.0):
                raise ModelError("recent_accuracy out of range")
            self.recent_accuracy = None if ra is None else float(ra)
            seen = payload.get("auto_seen", [])
            if not isinstance(seen, list) or not all(isinstance(x, int) and not isinstance(x, bool) for x in seen):
                raise ModelError("auto_seen is malformed")
            self._auto_seen.clear(); self._auto_seen_set.clear()
            for x in seen[-self._auto_seen.maxlen:]:
                self._auto_seen.append(x)
                self._auto_seen_set.add(x)
        except KeyError as exc:
            raise ModelError(f"model file is missing {exc}") from exc
        self.seeded = True
