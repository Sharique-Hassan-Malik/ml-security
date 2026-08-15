"""Detectors for injected instructions, and an honest account of their limits.

Each detector returns a score in [0, 1] with a reason. They are separate because
they fail differently, and a firewall that reports which one fired is one you
can tune; a single opaque score is one you can only turn off.

The framing that matters more than any individual pattern: **prompt injection is
not a string-matching problem.** There is no set of regexes that separates "an
instruction" from "text that discusses instructions", because natural language
has no delimiter between data and code. Every detector here is a heuristic with
a measurable false-positive rate, and the corpus deliberately includes hard
negatives -- security documentation, papers about injection, bug reports quoting
attacks -- that a naive keyword matcher flags every time.

That is why `firewall.py` treats detection as *defence in depth* and puts the
load-bearing controls elsewhere: never letting retrieved content reach the
instruction channel, allowlisting tool calls, and filtering output. Those are
structural and hold regardless of what the text says.
"""

from __future__ import annotations

import base64
import binascii
import math
import re
import unicodedata
from dataclasses import dataclass, field


@dataclass
class Signal:
    name: str
    score: float
    reason: str
    spans: list[tuple[int, int]] = field(default_factory=list)

    def __str__(self) -> str:
        return f"{self.name}={self.score:.2f} ({self.reason})"


# ---------------------------------------------------------------------------
# 1. instruction override
# ---------------------------------------------------------------------------

# Phrases that attempt to void prior context. The verbs matter more than the
# nouns: "ignore", "disregard", "forget" plus a reference to earlier instruction.
OVERRIDE_PATTERNS = [
    (r"\b(ignore|disregard|forget|discard|override)\b[^.\n]{0,40}\b"
     r"(previous|prior|above|earlier|preceding|all)\b[^.\n]{0,30}\b"
     r"(instruction|prompt|rule|direction|command|context|message)s?\b", 0.9),
    (r"\bnew\s+(instruction|rule|directive|task)s?\s*:", 0.7),
    (r"\byou\s+are\s+now\b[^.\n]{0,60}", 0.55),
    (r"\bfrom\s+now\s+on\b[^.\n]{0,60}\byou\b", 0.5),
    (r"\b(system|developer)\s*(prompt|message)\s*[:>]", 0.6),
    (r"\brepeat\b[^.\n]{0,30}\b(system prompt|instructions|above)\b", 0.75),
    (r"\b(reveal|show|print|output|display)\b[^.\n]{0,30}\b"
     r"(system prompt|your instructions|initial prompt|the prompt above)\b", 0.85),
]

# Fake conversation structure: an attacker closing the "user" turn and opening a
# "system" one. Real users do not write these.
ROLE_MARKERS = [
    (r"^\s*(###\s*)?(system|assistant|user)\s*:", 0.5),
    (r"<\|(im_start|im_end|system|endoftext)\|>", 0.9),
    (r"\[/?INST\]|<<SYS>>|\[/?SYSTEM\]", 0.85),
    (r"</?(system|instructions?)>", 0.6),
]

JAILBREAK_PATTERNS = [
    (r"\bDAN\b|\bdo anything now\b", 0.8),
    (r"\b(developer|god|debug|admin|root)\s+mode\b", 0.6),
    (r"\bpretend\b[^.\n]{0,40}\b(no|without)\b[^.\n]{0,30}\b(restriction|rule|filter|guideline)s?\b", 0.8),
    (r"\bhypothetically\b[^.\n]{0,50}\b(you|would you)\b[^.\n]{0,40}\b(could|able|allowed)\b", 0.4),
    (r"\bfor (educational|research) purposes only\b", 0.3),
    (r"\bjailbreak\b", 0.5),
    (r"\byour (safety|content) (policy|filter|guidelines?) (is|are) (now )?(disabled|off|suspended)\b", 0.9),
]

# Text that *discusses* injection rather than performing it. These are the hard
# negatives, and without them any detector looks excellent.
DISCUSSION_MARKERS = [
    r"\bprompt injection\b", r"\bthreat model\b", r"\battack (surface|vector)\b",
    r"\bmitigat\w+\b", r"\bvulnerab\w+\b", r"\bCVE-\d{4}-\d+\b",
    r"\bthis (paper|post|article|section)\b", r"\bfor example[,:]", r"\be\.g\.[,:]?\s",
    r"\bdetect(or|ion)\b", r"\bdefen[cs]e\b", r"\bresearcher", r"\bsecurity (team|review)\b",
]


def _search(text: str, patterns: list[tuple[str, float]]) -> tuple[float, list[str], list[tuple[int, int]]]:
    best, reasons, spans = 0.0, [], []
    for pattern, weight in patterns:
        for match in re.finditer(pattern, text, re.IGNORECASE | re.MULTILINE):
            best = max(best, weight)
            reasons.append(match.group(0)[:60].strip())
            spans.append(match.span())
    return best, reasons, spans


def discussion_discount(text: str) -> float:
    """How much this reads like writing *about* attacks rather than an attack.

    A discount, not a veto. Attackers can and do wrap payloads in academic
    framing, so this lowers a score rather than clearing it -- which is exactly
    the precision/recall dial the evaluation measures.
    """
    hits = sum(1 for pattern in DISCUSSION_MARKERS if re.search(pattern, text, re.IGNORECASE))
    return min(0.45, 0.15 * hits)


def detect_override(text: str) -> Signal:
    score, reasons, spans = _search(text, OVERRIDE_PATTERNS)
    if score:
        score = max(0.0, score - discussion_discount(text))
    return Signal("instruction_override", score,
                  "; ".join(reasons[:3]) or "no override phrasing", spans)


def detect_role_confusion(text: str) -> Signal:
    score, reasons, spans = _search(text, ROLE_MARKERS)
    return Signal("role_confusion", score,
                  "; ".join(reasons[:3]) or "no role markers", spans)


def detect_jailbreak(text: str) -> Signal:
    score, reasons, spans = _search(text, JAILBREAK_PATTERNS)
    if score:
        score = max(0.0, score - discussion_discount(text))
    return Signal("jailbreak", score, "; ".join(reasons[:3]) or "no jailbreak phrasing", spans)


# ---------------------------------------------------------------------------
# 2. obfuscation
# ---------------------------------------------------------------------------

ZERO_WIDTH = "".join(chr(c) for c in (0x200B, 0x200C, 0x200D, 0x2060, 0xFEFF))
BIDI_CONTROLS = "".join(chr(c) for c in (0x202A, 0x202B, 0x202C, 0x202D, 0x202E, 0x2066, 0x2067, 0x2068, 0x2069))


def detect_obfuscation(text: str) -> Signal:
    """Invisible characters, bidi overrides and homoglyphs.

    Present in essentially no legitimate prose and cheap to check, which makes
    this the highest-precision detector here. Bidi overrides in particular have
    exactly one use in a prompt: making rendered text differ from parsed text.
    """
    reasons, score = [], 0.0
    zero_width = sum(text.count(c) for c in ZERO_WIDTH)
    if zero_width:
        score = max(score, 0.7)
        reasons.append(f"{zero_width} zero-width character(s)")
    bidi = sum(text.count(c) for c in BIDI_CONTROLS)
    if bidi:
        score = max(score, 0.95)
        reasons.append(f"{bidi} bidirectional override(s)")

    # Latin text carrying Cyrillic or Greek lookalikes.
    scripts: dict[str, int] = {}
    for char in text:
        if char.isalpha():
            try:
                name = unicodedata.name(char).split()[0]
            except ValueError:
                continue
            scripts[name] = scripts.get(name, 0) + 1
    if len(scripts) > 1 and "LATIN" in scripts:
        foreign = sum(v for k, v in scripts.items() if k != "LATIN")
        total = sum(scripts.values())
        if total and 0 < foreign / total < 0.35:
            score = max(score, 0.6)
            reasons.append(f"mixed scripts: {sorted(scripts)}")
    return Signal("obfuscation", score, "; ".join(reasons) or "no obfuscation")


def detect_encoded_payload(text: str) -> Signal:
    """Base64 or hex that decodes to instruction-shaped text.

    Decoding before matching is the point: a detector that only reads the
    surface form is defeated by one `base64.b64encode`.
    """
    reasons, score = [], 0.0
    for candidate in re.findall(r"[A-Za-z0-9+/]{24,}={0,2}", text):
        try:
            decoded = base64.b64decode(candidate + "=" * (-len(candidate) % 4), validate=True)
            decoded_text = decoded.decode("utf-8", errors="strict")
        except (binascii.Error, UnicodeDecodeError, ValueError):
            continue
        if not decoded_text.isprintable():
            continue
        inner = max(detect_override(decoded_text).score, detect_jailbreak(decoded_text).score)
        if inner > 0:
            score = max(score, min(1.0, inner + 0.1))
            reasons.append(f"base64 decodes to: {decoded_text[:50]!r}")

    for candidate in re.findall(r"(?:[0-9a-fA-F]{2}[\s:]?){12,}", text):
        try:
            decoded_text = bytes.fromhex(re.sub(r"[\s:]", "", candidate)).decode("utf-8")
        except (ValueError, UnicodeDecodeError):
            continue
        if max(detect_override(decoded_text).score, detect_jailbreak(decoded_text).score) > 0:
            score = max(score, 0.85)
            reasons.append(f"hex decodes to: {decoded_text[:50]!r}")
    return Signal("encoded_payload", score, "; ".join(reasons[:2]) or "no encoded instructions")


def detect_delimiter_injection(text: str) -> Signal:
    """Attempts to close a delimiter the application uses to fence untrusted text.

    Only meaningful when the application actually fences input, which is why
    `Firewall` takes the delimiter as configuration rather than guessing.
    """
    reasons, score = [], 0.0
    for pattern, weight in [
        (r'"""|\'\'\'|```', 0.35),
        (r"</?(document|context|data|input|untrusted)>", 0.6),
        (r"-{3,}\s*(end|stop)\s+(of\s+)?(document|context|input)", 0.7),
    ]:
        if re.search(pattern, text, re.IGNORECASE):
            score = max(score, weight)
            reasons.append(pattern)
    return Signal("delimiter_injection", score, "; ".join(reasons[:2]) or "no delimiter escape")


def shannon_entropy(text: str) -> float:
    if not text:
        return 0.0
    counts: dict[str, int] = {}
    for char in text:
        counts[char] = counts.get(char, 0) + 1
    total = len(text)
    return -sum((n / total) * math.log2(n / total) for n in counts.values())


ALL_DETECTORS = (
    detect_override,
    detect_role_confusion,
    detect_jailbreak,
    detect_obfuscation,
    detect_encoded_payload,
    detect_delimiter_injection,
)


def scan(text: str) -> list[Signal]:
    return [detector(text) for detector in ALL_DETECTORS]
