"""Input normalization.

Attackers hide instructions behind Unicode tricks, zero-width characters,
homoglyphs, leetspeak, HTML entities and base64 blobs. This module folds an
input into canonical forms that downstream matchers can inspect, and it
records how much obfuscation it had to undo. Heavy obfuscation is itself a
signal: honest users rarely write in zero-width Cyrillic base64.
"""

from __future__ import annotations

import base64
import binascii
import html
import re
import unicodedata
from dataclasses import dataclass, field
from typing import Dict, List

# Characters that render as nothing but break naive pattern matching.
_ZERO_WIDTH = {
    "\u200b",  # zero width space
    "\u200c",  # zero width non-joiner
    "\u200d",  # zero width joiner
    "\u200e",  # left-to-right mark
    "\u200f",  # right-to-left mark
    "\u2060",  # word joiner
    "\u2061",
    "\u2062",
    "\u2063",
    "\u2064",
    "\u180e",  # mongolian vowel separator
    "\ufeff",  # byte order mark
    "\u00ad",  # soft hyphen
    "\u202a",  # bidi embedding and overrides
    "\u202b",
    "\u202c",
    "\u202d",
    "\u202e",
    "\u2066",
    "\u2067",
    "\u2068",
    "\u2069",
}

# Common confusables that survive NFKC. Keys are look-alikes, values are ASCII.
_HOMOGLYPHS: Dict[str, str] = {
    "\u0430": "a",  # Cyrillic a
    "\u0435": "e",  # Cyrillic ie
    "\u043e": "o",  # Cyrillic o
    "\u0440": "p",  # Cyrillic er
    "\u0441": "c",  # Cyrillic es
    "\u0443": "y",  # Cyrillic u
    "\u0445": "x",  # Cyrillic ha
    "\u0456": "i",  # Cyrillic byelorussian-ukrainian i
    "\u0458": "j",  # Cyrillic je
    "\u04bb": "h",  # Cyrillic shha
    "\u0455": "s",  # Cyrillic dze
    "\u0410": "A",
    "\u0412": "B",
    "\u0415": "E",
    "\u041a": "K",
    "\u041c": "M",
    "\u041d": "H",
    "\u041e": "O",
    "\u0420": "P",
    "\u0421": "C",
    "\u0422": "T",
    "\u0425": "X",
    "\u03b1": "a",  # Greek alpha
    "\u03bf": "o",  # Greek omicron
    "\u03c1": "p",  # Greek rho
    "\u03bd": "v",  # Greek nu
    "\u0391": "A",
    "\u0392": "B",
    "\u0395": "E",
    "\u0397": "H",
    "\u0399": "I",
    "\u039a": "K",
    "\u039c": "M",
    "\u039d": "N",
    "\u039f": "O",
    "\u03a1": "P",
    "\u03a4": "T",
    "\u03a5": "Y",
    "\u03a7": "X",
    "\u0561": "a",  # Armenian ayb
    "\u0585": "o",  # Armenian oh
    "\u2010": "-",
    "\u2011": "-",
    "\u2012": "-",
    "\u2013": "-",
    "\u2014": "-",
    "\u2018": "'",
    "\u2019": "'",
    "\u201c": '"',
    "\u201d": '"',
}

_LEET: Dict[str, str] = {
    "0": "o",
    "1": "i",
    "3": "e",
    "4": "a",
    "5": "s",
    "7": "t",
    "@": "a",
    "$": "s",
    "!": "i",
    "|": "l",
    "+": "t",
}

_BASE64_RUN = re.compile(r"(?<![A-Za-z0-9+/=])[A-Za-z0-9+/]{24,}={0,2}(?![A-Za-z0-9+/=])")
_HEX_RUN = re.compile(r"(?<![0-9A-Fa-f])(?:[0-9A-Fa-f]{2}){12,}(?![0-9A-Fa-f])")
_WS = re.compile(r"\s+")
_SPACED_LETTERS = re.compile(r"\b(?:[A-Za-z][ .\-_*]){4,}[A-Za-z]\b")
_MAX_DECODE_DEPTH = 2


@dataclass
class NormalizedText:
    """Canonical views of one input plus obfuscation signals."""

    original: str
    canonical: str
    folded: str
    decoded_payloads: List[str] = field(default_factory=list)
    signals: Dict[str, int] = field(default_factory=dict)

    @property
    def compact(self) -> str:
        """Folded text with all whitespace and punctuation removed.

        Used for phrase matching when letters were spaced or punctuated
        apart, which destroys word boundaries.
        """
        return re.sub(r"[^a-z0-9]", "", self.folded)

    @property
    def views(self) -> List[str]:
        """Every string a matcher should look at."""
        seen = []
        for v in [self.canonical, self.folded, *self.decoded_payloads]:
            if v and v not in seen:
                seen.append(v)
        return seen

    @property
    def obfuscation_score(self) -> float:
        """0.0 to 1.0. Rough measure of how much had to be undone."""
        s = self.signals
        weight = (
            0.15 * min(s.get("zero_width", 0), 4)
            + 0.10 * min(s.get("homoglyph", 0), 5)
            + 0.30 * min(s.get("decoded_payload", 0), 2)
            + 0.40 * min(s.get("spaced_letters", 0), 2)
            + 0.10 * min(s.get("control_chars", 0), 3)
            + 0.15 * min(s.get("bidi_override", 0), 2)
        )
        return min(1.0, weight)


def _strip_invisible(text: str) -> tuple[str, int, int, int]:
    zero_width = 0
    control = 0
    bidi = 0
    out = []
    for ch in text:
        if ch in _ZERO_WIDTH:
            zero_width += 1
            if ch in "\u202a\u202b\u202c\u202d\u202e\u2066\u2067\u2068\u2069":
                bidi += 1
            continue
        cat = unicodedata.category(ch)
        if cat in ("Cc", "Cf") and ch not in "\n\t\r":
            control += 1
            continue
        out.append(ch)
    return "".join(out), zero_width, control, bidi


def _fold_homoglyphs(text: str) -> tuple[str, int]:
    count = 0
    out = []
    for ch in text:
        repl = _HOMOGLYPHS.get(ch)
        if repl is not None:
            count += 1
            out.append(repl)
        else:
            out.append(ch)
    return "".join(out), count


def _fold_leet(text: str) -> str:
    # Only fold digits and symbols that sit inside alphabetic runs, so that
    # prices, dates and IDs are left alone.
    chars = list(text)
    n = len(chars)
    for i, ch in enumerate(chars):
        if ch not in _LEET:
            continue
        prev_alpha = i > 0 and chars[i - 1].isalpha()
        next_alpha = i + 1 < n and chars[i + 1].isalpha()
        if prev_alpha or next_alpha:
            chars[i] = _LEET[ch]
    return "".join(chars)


def _collapse_spaced_letters(text: str) -> tuple[str, int]:
    count = 0

    def _join(m: re.Match) -> str:
        nonlocal count
        count += 1
        return re.sub(r"[ .\-_*]", "", m.group(0))

    return _SPACED_LETTERS.sub(_join, text), count


def _printable_ratio(s: str) -> float:
    if not s:
        return 0.0
    good = sum(1 for c in s if c.isprintable() or c in "\n\t\r")
    return good / len(s)


def _try_decode_base64(blob: str) -> str | None:
    padded = blob + "=" * (-len(blob) % 4)
    try:
        raw = base64.b64decode(padded, validate=True)
    except (binascii.Error, ValueError):
        return None
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError:
        return None
    if len(text) < 8 or _printable_ratio(text) < 0.9:
        return None
    if not re.search(r"[A-Za-z]{3,}", text):
        return None
    return text


def _try_decode_hex(blob: str) -> str | None:
    try:
        raw = bytes.fromhex(blob)
        text = raw.decode("utf-8")
    except (ValueError, UnicodeDecodeError):
        return None
    if len(text) < 8 or _printable_ratio(text) < 0.9:
        return None
    if not re.search(r"[A-Za-z]{3,}", text):
        return None
    return text


def _extract_payloads(text: str, depth: int = 0) -> List[str]:
    if depth >= _MAX_DECODE_DEPTH:
        return []
    found: List[str] = []
    for m in _BASE64_RUN.finditer(text):
        decoded = _try_decode_base64(m.group(0))
        if decoded:
            found.append(decoded)
            found.extend(_extract_payloads(decoded, depth + 1))
    for m in _HEX_RUN.finditer(text):
        decoded = _try_decode_hex(m.group(0))
        if decoded:
            found.append(decoded)
            found.extend(_extract_payloads(decoded, depth + 1))
    return found


def normalize(text: str, max_chars: int = 200_000) -> NormalizedText:
    """Produce canonical views of ``text``.

    ``canonical`` preserves case and is safe to show to humans.
    ``folded`` is lowercase, leet-folded and whitespace-collapsed, for matching.
    ``decoded_payloads`` holds any base64 or hex that decoded to readable text.
    """
    if not isinstance(text, str):
        text = str(text)
    if len(text) > max_chars:
        text = text[:max_chars]

    signals: Dict[str, int] = {}

    step = unicodedata.normalize("NFKC", text)
    step = html.unescape(step)
    step, zero_width, control, bidi = _strip_invisible(step)
    step, homoglyphs = _fold_homoglyphs(step)
    canonical = _WS.sub(" ", step).strip()

    folded, spaced = _collapse_spaced_letters(canonical.lower())
    folded = _fold_leet(folded)
    folded = _WS.sub(" ", folded).strip()

    payloads = []
    for p in _extract_payloads(canonical):
        p_norm = unicodedata.normalize("NFKC", p)
        p_norm, _, _, _ = _strip_invisible(p_norm)
        p_norm, _ = _fold_homoglyphs(p_norm)
        payloads.append(_WS.sub(" ", p_norm.lower()).strip())

    if zero_width:
        signals["zero_width"] = zero_width
    if control:
        signals["control_chars"] = control
    if bidi:
        signals["bidi_override"] = bidi
    if homoglyphs:
        signals["homoglyph"] = homoglyphs
    if spaced:
        signals["spaced_letters"] = spaced
    if payloads:
        signals["decoded_payload"] = len(payloads)

    return NormalizedText(
        original=text,
        canonical=canonical,
        folded=folded,
        decoded_payloads=payloads,
        signals=signals,
    )
