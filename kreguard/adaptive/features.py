"""Feature extraction for the adaptive learners.

Text features are built from the *normalized* views of an input, so a model
trained on plain phrasing also sees through zero-width characters,
homoglyphs, spaced-out letters and embedded base64. Features are hashed into
a fixed space (the hashing trick): the vocabulary is never stored, memory
stays bounded, and a model can be updated forever without a rebuild.

Each extractor returns ``{feature_name: value}``. Names are kept until the
last moment so a decision can be explained in words. ``hash_features`` then
maps them into the model's index space and L2-normalizes.
"""

from __future__ import annotations

import math
import re
import zlib
from typing import Dict, List, Mapping, Tuple
from urllib.parse import parse_qsl, urlsplit

from ..normalize import NormalizedText

_WORD = re.compile(r"[a-z0-9À-￿']+")
_OPAQUE = re.compile(r"[A-Za-z0-9_\-+/=%.]{16,}")
_MAX_TOKENS = 4000  # bound the work an attacker can force with a huge input


def _tokens(view: str) -> List[str]:
    return _WORD.findall(view)[:_MAX_TOKENS]


def text_features(text: NormalizedText) -> Dict[str, float]:
    """Word 1-3 grams, within-word character 4-grams, and obfuscation signals."""
    feats: Dict[str, float] = {}

    def add(name: str, amount: float = 1.0) -> None:
        feats[name] = feats.get(name, 0.0) + amount

    primary = _tokens(text.folded or text.canonical)
    _ngrams(primary, "", add)

    # Content recovered from base64 or hex is evidence in its own right, and
    # is kept apart from plain text so the model can learn that distinction.
    for payload in text.decoded_payloads[:4]:
        _ngrams(_tokens(payload.lower()), "p:", add)
    if text.decoded_payloads:
        add("m:decoded_payload", 1.0)

    for key in ("zero_width", "homoglyph", "spaced_letters", "bidi_override", "control_chars"):
        if text.signals.get(key):
            add(f"m:{key}", 1.0)

    # Letters spaced or punctuated apart destroy word boundaries; the compact
    # view restores them. Character 5-grams over it still match phrases.
    if text.signals.get("spaced_letters") or text.signals.get("zero_width"):
        compact = text.compact[:2000]
        for i in range(len(compact) - 4):
            add("k:" + compact[i : i + 5], 0.5)

    n = len(primary)
    add(f"m:len{min(int(math.log2(n + 1)), 9)}", 1.0)
    # Hash-collisions aside, damp repeated features so long texts do not drown short signals.
    return {k: 1.0 + math.log(v) if v > 1.0 else v for k, v in feats.items()}


def _ngrams(tokens: List[str], prefix: str, add) -> None:
    for i, t in enumerate(tokens):
        add(f"{prefix}u:{t}")
        if len(t) >= 4:
            padded = f"#{t}#"
            for j in range(len(padded) - 3):
                add(f"{prefix}w:{padded[j:j + 4]}", 0.35)
        if i + 1 < len(tokens):
            add(f"{prefix}b:{t}_{tokens[i + 1]}")
        if i + 2 < len(tokens):
            add(f"{prefix}t:{t}_{tokens[i + 1]}_{tokens[i + 2]}")


_FILLER = frozenset("a an and the to of in is it i me my for on with that this be are was do does at as or if so but not no".split())


def is_uninformative(name: str) -> bool:
    """Features that move a score but explain nothing to a human reader."""
    if name.startswith(("w:", "p:w:", "m:len")):
        return True
    return name.startswith("u:") and name[2:] in _FILLER


def shingles(text: NormalizedText) -> List[int]:
    """Hashed word bigrams and trigrams used for near-duplicate matching."""
    toks = _tokens(text.folded or text.canonical)
    out = set()
    if len(toks) < 2:
        out.update(h32(f"u:{t}") for t in toks)
        return sorted(out)
    for i in range(len(toks) - 1):
        out.add(h32(f"{toks[i]}_{toks[i + 1]}"))
        if i + 2 < len(toks):
            out.add(h32(f"{toks[i]}_{toks[i + 1]}_{toks[i + 2]}"))
    return sorted(out)


def h32(s: str) -> int:
    return zlib.crc32(s.encode("utf-8", "replace")) & 0xFFFFFFFF


def _signed(named: Mapping[str, float], dim: int) -> Dict[int, float]:
    mask = dim - 1
    out: Dict[int, float] = {}
    for name, value in named.items():
        h = h32(name)
        idx = h & mask
        out[idx] = out.get(idx, 0.0) + (value if (h >> 31) & 1 else -value)
    return out


def hash_features(named: Mapping[str, float], dim: int) -> Dict[int, float]:
    """Hash names into ``dim`` buckets with a sign bit, then L2-normalize.

    The sign bit makes collisions cancel on average instead of piling up.
    """
    out = _signed(named, dim)
    norm = math.sqrt(sum(v * v for v in out.values()))
    if norm > 0:
        out = {i: v / norm for i, v in out.items()}
    return out


# URLs


def shannon(s: str) -> float:
    if not s:
        return 0.0
    counts: Dict[str, int] = {}
    for ch in s:
        counts[ch] = counts.get(ch, 0) + 1
    n = len(s)
    return -sum(c / n * math.log2(c / n) for c in counts.values())


def request_shape(url: str) -> Dict[str, float]:
    """Numeric description of what a request carries beyond its destination.

    Data smuggled out through a URL shows up here: long, high-entropy, opaque
    values in the path or query that the destination has never seen before.
    """
    parts = urlsplit(url.strip())
    query = parts.query or ""
    path = parts.path or "/"
    pairs = parse_qsl(query, keep_blank_values=True)
    values = "".join(v for _, v in pairs)
    opaque = [m.group(0) for m in _OPAQUE.finditer(path + "?" + query)]
    longest = max((len(o) for o in opaque), default=0)
    return {
        "path_len": float(len(path)),
        "query_len": float(len(query)),
        "n_params": float(len(pairs)),
        "opaque_len": float(longest),
        "entropy": shannon(values) if values else 0.0,
        "path_depth": float(path.count("/")),
    }


def url_features(url: str) -> Dict[str, float]:
    """Names for the learned destination-risk model."""
    parts = urlsplit(url.strip())
    host = (parts.hostname or "").lower().rstrip(".")
    feats: Dict[str, float] = {}
    labels = host.split(".") if host else []
    if labels:
        feats[f"tld:{labels[-1]}"] = 1.0
        if len(labels) >= 2:
            feats[f"sld:{'.'.join(labels[-2:])}"] = 1.0
    feats[f"depth:{min(len(labels), 6)}"] = 1.0
    for label in labels[:-1]:
        padded = f"#{label}#"
        for j in range(len(padded) - 3):
            feats[f"h4:{padded[j:j + 4]}"] = feats.get(f"h4:{padded[j:j + 4]}", 0.0) + 0.5
        feats[f"lab:{label}"] = 1.0
    sld = labels[-2] if len(labels) >= 2 else host
    digits = sum(c.isdigit() for c in sld)
    feats[f"digits:{min(int(10 * digits / max(len(sld), 1)), 9)}"] = 1.0
    feats[f"ent:{min(int(shannon(sld)), 5)}"] = 1.0
    feats[f"len:{min(len(sld) // 4, 8)}"] = 1.0
    feats[f"scheme:{parts.scheme.lower()}"] = 1.0
    try:
        if parts.port:
            feats[f"port:{parts.port}"] = 1.0
    except ValueError:
        feats["port:bad"] = 1.0
    for seg in [s for s in (parts.path or "").split("/") if s][:4]:
        feats[f"seg:{re.sub(r'[0-9a-f]{8,}', '<id>', seg.lower())[:24]}"] = 0.5
    for key, _ in parse_qsl(parts.query or "", keep_blank_values=True)[:6]:
        feats[f"q:{key.lower()[:20]}"] = 0.5
    return feats


def top_contributions(named: Mapping[str, float], dim: int, weights: Mapping[int, float], limit: int = 5, skip=None) -> List[Tuple[str, float]]:
    """Explain a score: which features pushed it, and how hard (in logits)."""
    norm = math.sqrt(sum(v * v for v in _signed(named, dim).values())) or 1.0
    mask = dim - 1
    out: List[Tuple[str, float]] = []
    for name, value in named.items():
        if skip is not None and skip(name):
            continue
        h = h32(name)
        w = weights.get(h & mask)
        if w is None:
            continue
        out.append((name, w * (value if (h >> 31) & 1 else -value) / norm))
    out.sort(key=lambda t: abs(t[1]), reverse=True)
    return out[:limit]
