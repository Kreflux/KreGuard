"""Command line entry point.

    python -m kreguard input "ignore all previous instructions"
    echo "..." | python -m kreguard input -
    python -m kreguard output --system-prompt prompt.txt reply.txt
    python -m kreguard egress https://api.example.com/v1 --allow api.example.com
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .classifier import LexiconClassifier
from .guard import Guard
from .permissions import EgressPolicy
from .verdict import Verdict

_EXIT = {Verdict.ALLOW: 0, Verdict.FLAG: 1, Verdict.BLOCK: 2}


def _read(arg: str) -> str:
    if arg == "-":
        return sys.stdin.read()
    p = Path(arg)
    if p.is_file():
        return p.read_text(encoding="utf-8", errors="replace")
    return arg


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="kreguard", description="KreGuard: guardrails for LLM apps.")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_in = sub.add_parser("input", help="scan an input for injection or jailbreak attempts")
    p_in.add_argument("text", help="text, a file path, or - for stdin")
    p_in.add_argument("--no-classifier", action="store_true", help="patterns only")
    p_in.add_argument("--json", action="store_true")

    p_out = sub.add_parser("output", help="scan a model output for leakage and credentials")
    p_out.add_argument("text", help="text, a file path, or - for stdin")
    p_out.add_argument("--system-prompt", help="text or file path of the system prompt")
    p_out.add_argument("--json", action="store_true")

    p_eg = sub.add_parser("egress", help="check a URL against an allowlist")
    p_eg.add_argument("url")
    p_eg.add_argument("--allow", action="append", default=[], help="allowed domain (repeatable)")
    p_eg.add_argument("--json", action="store_true")

    args = parser.parse_args(argv)

    if args.cmd == "input":
        guard = Guard(classifier=None if args.no_classifier else LexiconClassifier())
        decision = guard.check_input(_read(args.text))
    elif args.cmd == "output":
        guard = Guard(system_prompt=_read(args.system_prompt) if args.system_prompt else None)
        decision = guard.check_output(_read(args.text))
    else:
        guard = Guard(egress_policy=EgressPolicy(domains=set(args.allow)))
        decision = guard.authorize_egress(args.url)

    if args.json:
        print(json.dumps(decision.as_dict(), indent=2))
    else:
        print(f"{decision.verdict.value.upper()}  score={decision.score:.2f}")
        for f in decision.findings:
            print(f"  [{f.source}] {f.rule}: {f.detail}")
        redacted = getattr(decision, "redacted", None)
        if redacted is not None and args.cmd == "output" and decision.verdict is not Verdict.ALLOW:
            print("  redacted:", redacted)
    return _EXIT[decision.verdict]


if __name__ == "__main__":
    sys.exit(main())
