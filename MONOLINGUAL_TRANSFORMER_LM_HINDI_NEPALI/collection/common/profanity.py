"""Vulgar/profane-word filtering for Hindi and Nepali text.

A whole-word (not substring) match against a per-language blocklist. Substring
matching is deliberately avoided: many common, entirely innocuous Devanagari words
contain short sequences that overlap with slur/profanity roots, so a naive substring
check produces heavy false positives on ordinary news prose. Whole-word matching on
whitespace/punctuation-split tokens avoids that at the cost of missing inflected forms
not explicitly listed -- acceptable for a coarse pre-training-corpus filter, which only
needs to catch documents where such language is a substantial, deliberate presence
(e.g. comment-spam, forum flame text), not a linguistics-grade profanity detector.

The blocklists here are intentionally short and conservative (unambiguous, high-severity
terms only) to avoid over-filtering: this is corpus hygiene for a pretraining corpus,
not a chat-safety classifier, and the assignment does not require a strict content
policy -- the goal is simply to keep clearly low-quality/abusive text (which correlates
with poor writing quality generally, e.g. comment spam and flame threads) out of the
training data.
"""

from __future__ import annotations

import re

#: Deliberately short and conservative -- unambiguous profanity/slur roots only, to
#: avoid false-positiving on ordinary vocabulary. Documents are flagged, not silently
#: rewritten, so a false positive costs one document, not a corrupted one.
_HINDI_BLOCKLIST = {
    "मादरचोद", "बहनचोद", "भोसड़ीके", "भोसड़ी", "रंडी", "चूतिया", "चुतिया",
    "गांडू", "गांड", "लौड़ा", "लौड़े", "साला कुत्ता", "हरामी",
}
_NEPALI_BLOCKLIST = {
    "मादरचोद", "बहिनीचोद", "रन्डी", "छक्का", "पुतुर्नी", "बेस्या",
    "कुकुरको छोरा", "हरामी",
}

_BLOCKLIST_BY_LANG = {"hindi": _HINDI_BLOCKLIST, "nepali": _NEPALI_BLOCKLIST}

#: Fraction of a document's words that must match the blocklist before it is dropped.
#: Nonzero (not "any match drops the doc") because a single occurrence inside an
#: otherwise-legitimate document (e.g. a news report quoting an insult, a dictionary
#: entry) is a normal part of a real corpus and not itself low-quality text -- only
#: documents where such language is pervasive get filtered.
MAX_PROFANITY_RATIO = 0.01
#: Absolute-count floor, so a short document isn't dropped by rounding (one hit in a
#: 40-word document is a 2.5% ratio, above the 1% threshold, but is still just one word).
MIN_PROFANITY_HITS_TO_DROP = 3

_WORD_RE = re.compile(r"\S+")


def contains_excessive_profanity(text: str, lang: str) -> bool:
    """True if ``text`` contains enough blocklisted words to be dropped.

    Args:
        text: Document text (already NFC-normalized upstream).
        lang: ``"hindi"`` or ``"nepali"`` -- selects the blocklist.
    """
    blocklist = _BLOCKLIST_BY_LANG.get(lang)
    if not blocklist:
        return False
    words = _WORD_RE.findall(text)
    if not words:
        return False
    # Strip common trailing punctuation so "हरामी।" still matches "हरामी".
    hits = sum(1 for w in words if w.strip("।,.!?\"'()[]") in blocklist)
    if hits < MIN_PROFANITY_HITS_TO_DROP:
        return False
    return hits / len(words) > MAX_PROFANITY_RATIO
