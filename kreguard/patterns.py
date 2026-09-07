"""Pattern based detection of injection and jailbreak attempts.

Rules are regular expressions with a weight in (0, 1]. Weights combine with
noisy-or so several weak signals can add up to a strong one, while a single
high-confidence rule can push an input over the block threshold alone.

Patterns are cheap, transparent and easy to bypass. They catch the low
effort majority and leave the ambiguous middle to the classifier and judge.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterable, List, Optional, Pattern, Sequence

from .normalize import NormalizedText
from .verdict import Decision, Finding, Verdict


@dataclass(frozen=True)
class Rule:
    name: str
    pattern: Pattern[str]
    weight: float
    category: str
    description: str = ""

    def __post_init__(self) -> None:
        if not (0.0 < self.weight <= 1.0):
            raise ValueError(f"rule {self.name}: weight must be in (0, 1]")


def _r(name: str, regex: str, weight: float, category: str, description: str = "") -> Rule:
    return Rule(name, re.compile(regex, re.IGNORECASE), weight, category, description)


DEFAULT_RULES: List[Rule] = [
    # Instruction override
    _r(
        "ignore_previous_instructions",
        r"\b(ignore|disregard|forget|override|bypass|discard)\b[\w\s,]{0,40}?\b(all|any|every|the|your|prior|previous|above|earlier|preceding|initial|original|system)\b[\w\s,]{0,30}?\b(instruction|prompt|rule|guideline|directive|command|constraint|policy|programming)s?\b",
        0.85,
        "instruction_override",
    ),
    _r(
        "new_instructions_marker",
        r"\b(new|updated|revised|real|actual|true|hidden)\s+(instructions?|rules?|directives?|system prompt)\s*[:\-]",
        0.6,
        "instruction_override",
    ),
    _r(
        "from_now_on",
        r"\b(from now on|starting now|henceforth|for the rest of (this|the) (conversation|session|chat))\b[\w\s,]{0,60}?\b(you (will|must|shall|are)|always|never|only)\b",
        0.5,
        "instruction_override",
    ),
    _r(
        "stop_being",
        r"\b(stop|quit|cease)\s+(being|acting as|pretending to be)\s+(an?\s+)?(ai|assistant|language model|chatbot|helpful)",
        0.55,
        "instruction_override",
    ),
    # Role hijack
    _r(
        "you_are_now",
        r"\byou are now\b[\w\s,]{0,40}?\b(an?|the|my|in)\b",
        0.5,
        "role_hijack",
    ),
    _r(
        "pretend_persona",
        r"\b(pretend|imagine|act|roleplay|role-play|behave)\b[\w\s,]{0,20}?\b(as if|as though|that|you are|you're|to be)\b[\w\s,]{0,40}?\b(no (rules|restrictions|limits|filters|guidelines)|unrestricted|uncensored|unfiltered|evil|jailbroken|without (any )?(restrictions|limits|filters|rules|ethics|guidelines|safety))",
        0.85,
        "role_hijack",
    ),
    _r(
        "known_jailbreak_personas",
        r"\b(do anything now|god mode|evil confidant|anti-?gpt|opposite mode|mongo tom|jailbroken|always intelligent and machiavellian)\b",
        0.7,
        "role_hijack",
    ),
    Rule(
        "jailbreak_acronyms",
        # Case sensitive on purpose: "Dan" is a name, "DAN" is a persona.
        re.compile(r"\b(DAN|STAN|DUDE|AIM|JailBreak)\b(?=[\s,.:!]|$)"),
        0.6,
        "role_hijack",
    ),
    _r(
        "developer_mode_mention",
        r"\b(developer|dev|debug|unrestricted|uncensored) mode\b",
        0.3,
        "role_hijack",
        "Weak alone. Combines with enable_mode or persona rules.",
    ),
    _r(
        "enable_mode",
        r"\b(enable|activate|enter|switch (to|into)|unlock)\b\s+(the\s+)?(developer|debug|unrestricted|unfiltered|uncensored|admin|root|sudo|maintenance|god)\s+(mode|access|privileges?)",
        0.75,
        "role_hijack",
    ),
    # Safety bypass
    _r(
        "explicit_jailbreak",
        r"\b(jailbreak|jail-break|prompt injection|prompt-injection)\b",
        0.55,
        "safety_bypass",
    ),
    _r(
        "no_restrictions",
        r"\b(without|no|zero|free (of|from)|regardless of)\s+(any\s+)?(ethical|moral|safety|content|legal)?\s*(restrictions?|limitations?|filters?|guidelines?|guardrails?|censorship|boundaries|constraints?|rules)\b",
        0.6,
        "safety_bypass",
    ),
    _r(
        "bypass_safety",
        r"\b(bypass|circumvent|disable|turn off|remove|evade|defeat|get around)\b[\w\s]{0,20}?\b(safety|content|moderation|ethical|security)?\s*(filters?|guardrails?|restrictions?|safeguards?|checks?|protections?|policy|policies)\b",
        0.75,
        "safety_bypass",
    ),
    _r(
        "hypothetical_framing",
        r"\b(hypothetically|in theory|for (a|my) (novel|story|screenplay|research|thesis|movie)|purely fictional|in a fictional (world|scenario|universe))\b[\w\s,]{0,80}?\b(how (to|would|do|can)|steps?|instructions?|recipe|explain)\b",
        0.3,
        "safety_bypass",
    ),
    _r(
        "answer_anyway",
        r"\b(you (must|have to|will) (answer|comply|obey|respond)|refusal is not an option|do not refuse|never refuse|you cannot refuse|you are not allowed to (refuse|decline))\b",
        0.6,
        "safety_bypass",
    ),
    # System prompt exfiltration
    _r(
        "reveal_system_prompt",
        r"\b(reveal|show|print|display|output|repeat|recite|leak|dump|tell me|give me|write out|copy|paste|echo|share|expose|disclose)\b[\w\s,]{0,40}?\b(your|the|its|all)\s+(system|initial|original|hidden|secret|internal|full|complete|above|previous|first|entire|exact)\s+(prompt|instructions?|message|configuration|config|context|rules|directives?)",
        0.85,
        "prompt_exfiltration",
    ),
    _r(
        "what_are_your_instructions",
        r"\b(what|which)\s+(are|were|is|was)\s+(your|the)\s+(system|initial|original|hidden|secret|exact)\s+(prompt|instructions?|rules|directives?)",
        0.6,
        "prompt_exfiltration",
    ),
    _r(
        "everything_above",
        r"\b(repeat|print|output|echo|display|show)\b[\w\s]{0,20}?\b(everything|all( the)? text|the text|words?|content)\s+(above|before this|prior to this|that came before|preceding)",
        0.8,
        "prompt_exfiltration",
    ),
    _r(
        "start_with_you_are",
        r"\b(start|begin)\s+(your\s+)?(response|answer|reply|output)\s+with\s+[\"']?(you are|system|as an ai|the system prompt)",
        0.6,
        "prompt_exfiltration",
    ),
    # Delimiter and template injection
    _r(
        "chat_template_tokens",
        r"(<\|?(im_start|im_end|system|user|assistant|endoftext|eot_id|start_header_id|end_header_id)\|?>|\[/?INST\]|<<SYS>>|<</SYS>>|\[/?SYSTEM\]|###\s*(system|instruction|human|assistant)\s*[:\n]|^\s*(system|assistant)\s*:\s)",
        0.8,
        "delimiter_injection",
    ),
    _r(
        "fake_section_markers",
        r"\b(begin|end|start)\s+(of\s+)?(system|admin|hidden|secret|developer|privileged)\s+(prompt|message|instructions?|section|block|override)\b",
        0.75,
        "delimiter_injection",
    ),
    _r(
        "authority_claim",
        r"\b(i am|i'm|this is|message from)\s+(the\s+|your\s+|an?\s+)?(developer|administrator|admin|openai|anthropic|google|system|root|creator|engineer|operator|owner)\b[\w\s,]{0,40}?\b(and|so|therefore|now|override|authorize|permit|allow|grant|command|order)\b",
        0.6,
        "delimiter_injection",
    ),
    # Indirect injection carried in documents and tool results
    _r(
        "ai_addressed_document",
        r"\b(ai|assistant|chatgpt|claude|gemini|copilot|llm|language model|agent)\s*[,:]?\s*(if you are reading this|reading this|note to|important instruction|attention)\b",
        0.7,
        "indirect_injection",
    ),
    _r(
        "hidden_instruction_comment",
        r"(<!--|/\*|//|#)\s*[\w\s]{0,20}?\b(ai|assistant|agent|model)\b[\w\s]{0,60}?\b(ignore|must|should|do not|don't|always|never|instruct|instructions?)\b",
        0.6,
        "indirect_injection",
    ),
    _r(
        "do_not_tell_user",
        r"\b(do not|don't|never)\s+(tell|inform|mention|reveal|show|alert|notify|let)\s+(the\s+)?(user|human|operator|them)\b",
        0.7,
        "indirect_injection",
    ),
    # Exfiltration and tool coercion
    _r(
        "exfil_via_url",
        r"\b(send|post|upload|transmit|forward|exfiltrate|submit|leak)\b[\w\s,]{0,40}?\b(to|at)\s+(https?://|ftp://|www\.|[\w\-]+\.(com|net|org|io|xyz|ru|cn|tk|top|dev|me|co)\b)",
        0.7,
        "exfiltration",
    ),
    _r(
        "markdown_image_exfil",
        r"!\[[^\]]*\]\(\s*https?://[^)\s]*[?&#][^)\s]*\)",
        0.6,
        "exfiltration",
        "Markdown image with query string: classic zero-click exfil channel.",
    ),
    _r(
        "embed_secrets_in_url",
        r"\b(append|include|add|encode|put|insert)\b[\w\s,]{0,40}?\b(conversation|chat|history|secrets?|keys?|tokens?|passwords?|credentials?|data|context|prompt)\b[\w\s,]{0,40}?\b(to|in|into|as)\s+(the\s+)?(url|link|query|parameter|image|request)",
        0.75,
        "exfiltration",
    ),
    _r(
        "tool_coercion",
        r"\b(call|invoke|run|execute|use|trigger)\s+(the\s+)?(tool|function|command|api|endpoint|shell|script)\b[\w\s,]{0,40}?\b(delete|drop|rm -rf|remove|transfer|send|wire|pay|purchase|buy|sudo|chmod|curl|wget)\b",
        0.6,
        "tool_coercion",
    ),
    _r(
        "shell_payload",
        r"(\brm\s+-rf\b|\bcurl\b[^\n]{0,80}\|\s*(ba)?sh\b|\bwget\b[^\n]{0,80}\|\s*(ba)?sh\b|/etc/(passwd|shadow)|\bnc\s+-e\b|\bbase64\s+-d\b\s*\|)",
        0.7,
        "tool_coercion",
    ),
    # Encoding requests intended to evade output filters
    _r(
        "encode_output",
        r"\b(respond|reply|answer|write|output|encode)\b[\w\s]{0,20}?\b(in|using|as|with)\s+(base64|rot13|hex|binary|morse|pig latin|leetspeak|reversed text|a cipher|caesar)\b",
        0.5,
        "evasion",
    ),
    _r(
        "split_and_recombine",
        r"\b(first|second|next|last)\s+(half|part|letter|character)s?\s+of\s+(each|every)\s+(word|line|sentence)\b",
        0.4,
        "evasion",
    ),
]


# Phrases matched against the compact view (no whitespace or punctuation).
# These catch "i g n o r e   a l l ..." and "i.g.n.o.r.e" style spacing.
COMPACT_PHRASES: List[Rule] = [
    _r("compact_ignore_instructions", r"(ignore|disregard|forget|override)(all|any|the|your)?(previous|prior|above|earlier|system)?(instructions|rules|prompt)", 0.75, "obfuscated"),
    _r("compact_system_prompt_exfil", r"(reveal|show|print|repeat|leak|dump|display)(me)?(your|the)?(system|hidden|secret|initial)(prompt|instructions)", 0.75, "obfuscated"),
    _r("compact_personas", r"(developermode|doanythingnow|jailbreak|norestrictions|withoutrestrictions|unrestrictedmode|godmode)", 0.7, "obfuscated"),
    _r("compact_you_are_now", r"youarenow(an?)?(unrestricted|uncensored|evil|free|dan)", 0.75, "obfuscated"),
]


class PatternScanner:
    """Scan normalized text against a rule set and produce a Decision."""

    def __init__(
        self,
        rules: Optional[Sequence[Rule]] = None,
        extra_rules: Iterable[Rule] = (),
        flag_threshold: float = 0.35,
        block_threshold: float = 0.8,
        obfuscation_weight: float = 0.5,
    ) -> None:
        base = list(DEFAULT_RULES if rules is None else rules)
        base.extend(extra_rules)
        names = [r.name for r in base]
        if len(names) != len(set(names)):
            raise ValueError("duplicate rule names")
        if not (0.0 <= flag_threshold < block_threshold <= 1.0):
            raise ValueError("thresholds must satisfy 0 <= flag < block <= 1")
        self.rules = base
        self.flag_threshold = flag_threshold
        self.block_threshold = block_threshold
        self.obfuscation_weight = obfuscation_weight

    @staticmethod
    def _noisy_or(weights: Iterable[float]) -> float:
        survive = 1.0
        for w in weights:
            survive *= 1.0 - w
        return 1.0 - survive

    def scan(self, text: NormalizedText) -> Decision:
        decision = Decision(verdict=Verdict.ALLOW, stage="patterns")
        hit_weights: List[float] = []
        seen_rules = set()

        for view_index, view in enumerate(text.views):
            in_payload = view_index >= 2
            for rule in self.rules:
                if rule.name in seen_rules:
                    continue
                m = rule.pattern.search(view)
                if not m:
                    continue
                seen_rules.add(rule.name)
                # A match found only after decoding an embedded payload is
                # more suspicious, not less: nobody base64-encodes small talk.
                weight = min(1.0, rule.weight * (1.15 if in_payload else 1.0))
                hit_weights.append(weight)
                snippet = m.group(0)
                if len(snippet) > 80:
                    snippet = snippet[:77] + "..."
                decision.findings.append(
                    Finding(
                        source="patterns",
                        rule=rule.name,
                        verdict=Verdict.FLAG,
                        detail=f"[{rule.category}] {snippet!r}"
                        + (" (in decoded payload)" if in_payload else ""),
                        score=weight,
                    )
                )

        if text.signals.get("spaced_letters") or text.signals.get("zero_width"):
            compact = text.compact
            for rule in COMPACT_PHRASES:
                if rule.name in seen_rules:
                    continue
                m = rule.pattern.search(compact)
                if not m:
                    continue
                seen_rules.add(rule.name)
                hit_weights.append(rule.weight)
                decision.findings.append(
                    Finding("patterns", rule.name, Verdict.FLAG, f"[{rule.category}] {m.group(0)!r} (compact view)", rule.weight)
                )

        obfuscation = text.obfuscation_score
        if obfuscation > 0:
            contribution = obfuscation * self.obfuscation_weight
            # Obfuscation only counts when something was actually found,
            # or when it is extreme on its own.
            if hit_weights or obfuscation >= 0.6:
                hit_weights.append(contribution)
                decision.findings.append(
                    Finding(
                        source="patterns",
                        rule="obfuscation",
                        verdict=Verdict.FLAG,
                        detail=", ".join(f"{k}={v}" for k, v in sorted(text.signals.items())),
                        score=contribution,
                    )
                )

        score = self._noisy_or(hit_weights)
        decision.score = score
        if score >= self.block_threshold:
            decision.verdict = Verdict.BLOCK
        elif score >= self.flag_threshold:
            decision.verdict = Verdict.FLAG
        else:
            decision.verdict = Verdict.ALLOW
        return decision
