"""Append-only JSON Lines audit log for guard decisions.

A guardrail you cannot inspect afterwards is a guardrail you cannot tune.
Each decision becomes one line: when, which check, the verdict, the rules
that fired. By default the log does not contain the text that was checked,
only its SHA-256 and length, because prompts and model replies routinely
carry personal data. Turn on ``include_text`` if you want the raw subject.

A failing disk never changes a verdict. The write error is reported once on
stderr and the decision stands.
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
import threading
import time
from pathlib import Path
from typing import IO, Any, Mapping, Optional, Union

from .verdict import Decision


class AuditLog:
    def __init__(
        self,
        path: Union[str, Path, None] = None,
        *,
        stream: Optional[IO[str]] = None,
        include_text: bool = False,
    ) -> None:
        if path is None and stream is None:
            raise ValueError("AuditLog needs a path or a stream")
        self.include_text = include_text
        self._lock = threading.Lock()
        self._warned = False
        self._owns_stream = False
        if stream is not None:
            self._stream: IO[str] = stream
        else:
            # Owner read/write only: the log describes what users asked.
            fd = os.open(str(path), os.O_WRONLY | os.O_APPEND | os.O_CREAT, 0o600)
            self._stream = os.fdopen(fd, "a", encoding="utf-8")
            self._owns_stream = True

    def record(
        self,
        kind: str,
        decision: Decision,
        subject: Optional[str] = None,
        display: Optional[str] = None,
        extra: Optional[Mapping[str, Any]] = None,
    ) -> None:
        """Write one decision. ``display`` is a caller-vetted safe summary of
        the subject (a hostname, a tool name) that is always logged."""
        entry: dict = {
            "ts": round(time.time(), 3),
            "kind": kind,
            "verdict": decision.verdict.value,
            "stage": decision.stage,
            "score": round(decision.score, 4),
            "rules": [f"{f.source}:{f.rule}" for f in decision.findings],
        }
        if subject is not None:
            entry["subject_sha256"] = hashlib.sha256(subject.encode("utf-8", "replace")).hexdigest()
            entry["subject_len"] = len(subject)
            if self.include_text:
                entry["subject"] = subject
        if display is not None:
            entry["display"] = display
        if extra:
            entry["extra"] = dict(extra)
        try:
            line = json.dumps(entry, ensure_ascii=False, separators=(",", ":"))
            with self._lock:
                self._stream.write(line + "\n")
                self._stream.flush()
        except Exception as exc:  # noqa: BLE001 - logging must not alter a verdict
            if not self._warned:
                self._warned = True
                print(f"kreguard: audit log write failed: {exc}", file=sys.stderr)

    def close(self) -> None:
        if self._owns_stream:
            try:
                self._stream.close()
            except Exception:  # noqa: BLE001
                pass
