"""Online logistic regression and safe model persistence.

The learner is deliberately small and fully inspectable: sparse weights over
hashed features, updated one example at a time. No dependencies, no random
numbers, no hidden state. The same inputs in the same order always produce
the same model.

Two choices matter for a guard that has to keep learning while attackers
change tactics:

* The AdaGrad accumulator decays (``rho < 1``). Plain AdaGrad shrinks its
  learning rate forever and eventually stops adapting. A decaying
  accumulator keeps a feature responsive to new evidence, so when the world
  drifts the weights can follow.
* Weights are clipped. A flood of crafted feedback cannot push any single
  feature to an arbitrarily large value.

Persistence treats the model file as untrusted input. It is versioned,
checksummed, written atomically with owner-only permissions, and validated
on load: every number must be finite and within bounds. A file that fails any
check is rejected, never half-loaded.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import tempfile
from pathlib import Path
from typing import Any, Dict, Mapping, Union

FORMAT_VERSION = 1
_EPS = 1e-8


class ModelError(ValueError):
    """A model file is corrupt, tampered with, or incompatible."""


def sigmoid(z: float) -> float:
    if z >= 0:
        e = math.exp(-min(z, 60.0))
        return 1.0 / (1.0 + e)
    e = math.exp(max(z, -60.0))
    return e / (1.0 + e)


class OnlineLogReg:
    """Sparse logistic regression trained by decayed-AdaGrad SGD."""

    def __init__(
        self,
        dim: int = 1 << 18,
        lr: float = 0.35,
        l2: float = 1e-5,
        rho: float = 0.99,
        clip: float = 6.0,
        max_features: int = 200_000,
    ) -> None:
        if dim < 16 or dim & (dim - 1):
            raise ValueError("dim must be a power of two >= 16")
        if not (0.0 < lr <= 5.0 and 0.0 <= l2 < 1.0 and 0.0 < rho <= 1.0 and clip > 0):
            raise ValueError("invalid hyperparameters")
        self.dim = dim
        self.lr = lr
        self.l2 = l2
        self.rho = rho
        self.clip = clip
        self.max_features = max_features
        self.w: Dict[int, float] = {}
        self.g2: Dict[int, float] = {}
        self.bias = 0.0
        self.bias_g2 = 0.0
        self.updates = 0

    # Inference

    def margin(self, feats: Mapping[int, float]) -> float:
        w = self.w
        z = self.bias
        for i, v in feats.items():
            wi = w.get(i)
            if wi is not None:
                z += wi * v
        return z

    def predict(self, feats: Mapping[int, float]) -> float:
        return sigmoid(self.margin(feats))

    # Training

    def update(self, feats: Mapping[int, float], label: int, weight: float = 1.0) -> float:
        """One SGD step. Returns the probability before the update."""
        p = self.predict(feats)
        err = (p - label) * weight
        w, g2, rho, lr, l2, clip = self.w, self.g2, self.rho, self.lr, self.l2, self.clip
        for i, v in feats.items():
            wi = w.get(i, 0.0)
            g = err * v + l2 * wi
            acc = rho * g2.get(i, 0.0) + g * g
            g2[i] = acc
            nw = wi - lr * g / (math.sqrt(acc) + _EPS)
            w[i] = max(-clip, min(clip, nw))
        gb = err
        self.bias_g2 = rho * self.bias_g2 + gb * gb
        self.bias = max(-clip, min(clip, self.bias - lr * gb / (math.sqrt(self.bias_g2) + _EPS)))
        self.updates += 1
        if len(w) > self.max_features:
            self._prune()
        return p

    def _prune(self) -> None:
        """Drop the least informative weights once over budget."""
        keep = int(self.max_features * 0.9)
        ranked = sorted(self.w, key=lambda i: abs(self.w[i]), reverse=True)[:keep]
        keep_set = set(ranked)
        self.w = {i: self.w[i] for i in keep_set}
        self.g2 = {i: self.g2[i] for i in keep_set if i in self.g2}

    # Serialization

    def to_dict(self) -> Dict[str, Any]:
        idx = sorted(self.w)
        return {
            "dim": self.dim,
            "lr": self.lr,
            "l2": self.l2,
            "rho": self.rho,
            "clip": self.clip,
            "max_features": self.max_features,
            "bias": self.bias,
            "bias_g2": self.bias_g2,
            "updates": self.updates,
            "idx": idx,
            "w": [round(self.w[i], 8) for i in idx],
            "g2": [round(self.g2.get(i, 0.0), 8) for i in idx],
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "OnlineLogReg":
        try:
            dim = _int(data["dim"], 16, 1 << 26, "dim")
            m = cls(
                dim=dim,
                lr=_num(data["lr"], 1e-6, 5.0, "lr"),
                l2=_num(data["l2"], 0.0, 0.999, "l2"),
                rho=_num(data["rho"], 1e-6, 1.0, "rho"),
                clip=_num(data["clip"], 1e-3, 100.0, "clip"),
                max_features=_int(data["max_features"], 1, 5_000_000, "max_features"),
            )
            idx, w, g2 = data["idx"], data["w"], data["g2"]
            if not (isinstance(idx, list) and isinstance(w, list) and isinstance(g2, list) and len(idx) == len(w) == len(g2)):
                raise ModelError("weight arrays are malformed")
            for i, wi, gi in zip(idx, w, g2):
                i = _int(i, 0, dim - 1, "feature index")
                m.w[i] = _num(wi, -m.clip, m.clip, "weight")
                m.g2[i] = _num(gi, 0.0, 1e9, "accumulator")
            m.bias = _num(data["bias"], -m.clip, m.clip, "bias")
            m.bias_g2 = _num(data["bias_g2"], 0.0, 1e9, "bias accumulator")
            m.updates = _int(data["updates"], 0, 1 << 62, "updates")
            return m
        except KeyError as exc:
            raise ModelError(f"model is missing field {exc}") from exc
        except ValueError as exc:
            if isinstance(exc, ModelError):
                raise
            raise ModelError(f"model hyperparameters are invalid: {exc}") from exc


def _num(value: Any, lo: float, hi: float, what: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ModelError(f"{what} is not a number")
    v = float(value)
    if not math.isfinite(v) or v < lo or v > hi:
        raise ModelError(f"{what} out of range")
    return v


def _int(value: Any, lo: int, hi: int, what: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < lo or value > hi:
        raise ModelError(f"{what} out of range")
    return value


# File handling


def _digest(payload: Mapping[str, Any]) -> str:
    blob = json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()


def save_json(path: Union[str, Path], kind: str, payload: Mapping[str, Any]) -> None:
    """Atomically write a checksummed model file readable only by its owner."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    doc = {"format": f"kreguard-{kind}", "version": FORMAT_VERSION, "sha256": _digest(payload), "payload": payload}
    fd, tmp = tempfile.mkstemp(prefix=p.name + ".", suffix=".tmp", dir=str(p.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            json.dump(doc, fh, allow_nan=False, separators=(",", ":"))
            fh.flush()
            os.fsync(fh.fileno())
        os.chmod(tmp, 0o600)
        os.replace(tmp, p)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def load_json(path: Union[str, Path], kind: str, max_bytes: int = 256 * 1024 * 1024) -> Dict[str, Any]:
    p = Path(path)
    try:
        if p.stat().st_size > max_bytes:
            raise ModelError(f"{p} is larger than {max_bytes} bytes")
        doc = json.loads(p.read_text(encoding="utf-8"), parse_constant=_reject_constant)
    except OSError as exc:
        raise ModelError(f"cannot read {p}: {exc}") from exc
    except (ValueError, RecursionError) as exc:
        if isinstance(exc, ModelError):
            raise
        raise ModelError(f"{p} is not valid JSON: {exc}") from exc
    if not isinstance(doc, dict) or doc.get("format") != f"kreguard-{kind}":
        raise ModelError(f"{p} is not a kreguard-{kind} file")
    if doc.get("version") != FORMAT_VERSION:
        raise ModelError(f"{p} has unsupported version {doc.get('version')!r}")
    payload = doc.get("payload")
    if not isinstance(payload, dict):
        raise ModelError(f"{p} has no payload")
    if doc.get("sha256") != _digest(payload):
        raise ModelError(f"{p} failed its integrity check (corrupt or edited)")
    return payload


def _reject_constant(name: str) -> Any:
    raise ModelError(f"JSON constant {name} is not allowed")

