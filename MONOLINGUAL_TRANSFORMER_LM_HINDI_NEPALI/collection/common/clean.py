"""Cleaning pipeline: Unicode normalization, quality filters, dedup, decontamination.

Every raw shard (news, Wikipedia, IndicCorp) goes through the same sequence:

1. NFC-normalize the text and strip control characters / stray zero-width marks.
2. Drop documents shorter than :data:`MIN_CHARS` -- too short to be a real article.
3. Drop documents whose Devanagari-letter ratio is too low or Latin-letter ratio too
   high -- catches pages that are mostly English boilerplate or navigation chrome.
4. Drop documents that are mostly one repeated line or n-gram -- catches template
   boilerplate (nav menus, disclaimer blocks, auto-generated listing pages) that
   passes the script filter cleanly since the repeated text is real Devanagari, just
   not informative running prose.
5. Drop documents with an excessive concentration of blocklisted profanity/slurs
   (see :mod:`.profanity`) -- coarse corpus hygiene, not a content policy: a single
   flagged word in an otherwise normal document (e.g. quoted in a news report) is not
   itself grounds for removal, only documents where such language is pervasive.
6. Drop documents matching an entry in the decontamination set (see :mod:`.decontam`)
   -- exact and near-duplicate matches against the Phase 3 synthetic reasoning-eval
   templates/entities, so no benchmark text leaks into the pretraining corpus.
7. Drop exact duplicates by SHA-256 of the normalized text, tracked across the whole
   corpus (all input directories together), not per-shard.
8. Drop near-duplicates via MinHash/LSH (see :mod:`.near_dedup`) -- catches
   wire-service reprints and near-identical crawl entries that exact hashing misses
   because they differ by a byline, dateline, or a few edited words.

Devanagari nukta letters (U+0958-U+095F) sit on Unicode's own composition-exclusion
list, so plain NFC already normalizes both the precomposed and decomposed spellings to
one form -- no extra nukta-specific handling is needed here; see
``tests/test_clean.py`` for a pinned regression case.
"""

from __future__ import annotations

import hashlib
import re
import shutil
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path

from .near_dedup import NearDedupFilter
from .profanity import contains_excessive_profanity
from .record import Record, ShardWriter, iter_records

MIN_CHARS = 200
MIN_DEVANAGARI_RATIO = 0.70
MAX_LATIN_RATIO = 0.15
#: A document is dropped as repetitive if the single most common line, or the single
#: most common word 3-gram, makes up more than this fraction of the document.
MAX_LINE_REPETITION_RATIO = 0.30
MAX_NGRAM_REPETITION_RATIO = 0.20

#: Control characters (excluding \n, \t) and zero-width space -- always noise.
_CONTROL_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f​]")
#: ZWNJ/ZWJ are only stripped when clearly stray (surrounded by whitespace, or
#: doubled), since they legitimately control conjunct rendering in Devanagari text.
_STRAY_ZW_RE = re.compile(r"(?:\s[‌‍]\s|[‌‍]{2,})")

_DEVANAGARI_RE = re.compile(r"[ऀ-ॿ]")
_LATIN_RE = re.compile(r"[A-Za-z]")


@dataclass
class CleanStats:
    seen: int = 0
    kept: int = 0
    dropped_short: int = 0
    dropped_script: int = 0
    dropped_repetitive: int = 0
    dropped_profanity: int = 0
    dropped_contaminated: int = 0
    dropped_duplicate: int = 0
    dropped_near_duplicate: int = 0
    output_bytes: int = 0
    per_method: dict[str, int] = field(default_factory=dict)


def normalize_text(text: str) -> str:
    """NFC-normalize and strip control/stray-zero-width noise."""
    text = unicodedata.normalize("NFC", text)
    text = _CONTROL_RE.sub("", text)
    text = _STRAY_ZW_RE.sub(" ", text)
    return text.strip()


def script_ratios(text: str) -> tuple[float, float]:
    """Return (devanagari_ratio, latin_ratio) over letter characters only."""
    letters = [c for c in text if c.isalpha()]
    if not letters:
        return 0.0, 0.0
    dev = sum(1 for c in letters if _DEVANAGARI_RE.match(c))
    lat = sum(1 for c in letters if _LATIN_RE.match(c))
    n = len(letters)
    return dev / n, lat / n


def is_repetitive(text: str) -> bool:
    """Flag documents dominated by one repeated line or word 3-gram.

    Catches templated boilerplate a script-ratio filter cannot see -- a page that is
    90% a repeated disclaimer block or nav-menu line is still valid Devanagari, just
    not informative running prose. Two independent signals, either one enough to
    flag: line-level (catches repeated blocks/rows) and 3-gram-level (catches
    repeated short phrases within otherwise-varied lines, e.g. a boilerplate share
    button phrase re-inserted after every paragraph).
    """
    lines = [l for l in text.split("\n") if l.strip()]
    if len(lines) >= 4:
        counts: dict[str, int] = {}
        for line in lines:
            counts[line] = counts.get(line, 0) + 1
        if max(counts.values()) / len(lines) > MAX_LINE_REPETITION_RATIO:
            return True

    words = text.split()
    if len(words) >= 30:
        ngram_counts: dict[tuple[str, str, str], int] = {}
        total_ngrams = len(words) - 2
        for i in range(total_ngrams):
            gram = (words[i], words[i + 1], words[i + 2])
            ngram_counts[gram] = ngram_counts.get(gram, 0) + 1
        if max(ngram_counts.values()) / total_ngrams > MAX_NGRAM_REPETITION_RATIO:
            return True

    return False


def passes_filters(text: str, lang: str | None = None) -> tuple[bool, str | None]:
    """Return (ok, reason_if_dropped).

    ``lang`` selects the profanity blocklist (see :mod:`.profanity`); omitted or
    unrecognized means the profanity check is skipped, not that the document is
    dropped -- callers that know their language should always pass it.
    """
    if len(text) < MIN_CHARS:
        return False, "short"
    dev_ratio, lat_ratio = script_ratios(text)
    if dev_ratio < MIN_DEVANAGARI_RATIO or lat_ratio > MAX_LATIN_RATIO:
        return False, "script"
    if is_repetitive(text):
        return False, "repetitive"
    if lang is not None and contains_excessive_profanity(text, lang):
        return False, "profanity"
    return True, None


#: Directory-name substrings whose documents get the MinHash/LSH near-dedup pass.
#: Deliberately excludes large public/downloaded corpora (IndicCorp, Sangraha) --
#: near-dedup's LSH index grows with every document it indexes (~5-11GB estimated at
#: this project's ~8.5M-document IndicCorp scale on a Python set, found by watching
#: RSS climb toward an OOM during an actual run), which does not fit this machine's
#: ~7GB RAM. Manual news + Wikipedia together are a few hundred thousand documents at
#: most, where the index stays a few hundred MB, and is also where near-duplicates
#:(wire-service reprints across outlets, near-identical Wikipedia stub variants)
#: actually matter most -- a broad web crawl like IndicCorp already gets its own
#: dedup pass upstream by its publishers, so exact-dedup here is enough for it.
NEAR_DEDUP_DIR_MARKERS = ("manual_news", "manual_wiki")


def _should_near_dedup(input_dir: Path) -> bool:
    return any(marker in input_dir.name for marker in NEAR_DEDUP_DIR_MARKERS)


def clean_corpus(
    input_dirs: list[Path],
    out_dir: Path,
    lang: str | None = None,
    seed_hashes: set[str] | None = None,
    contamination: "ContaminationSet | None" = None,
    near_dedup: bool = True,
) -> CleanStats:
    """Normalize, filter, decontaminate, and dedup every record under ``input_dirs``.

    Stage order matters for both correctness and cost: cheap per-document filters
    (length, script, repetition, profanity) run first so expensive later stages
    (decontamination's substring scan, MinHash near-dedup) only ever see documents
    that already passed everything cheaper -- never wasted on a document that would
    have been dropped anyway.

    Args:
        input_dirs: Raw shard directories to merge (news, wiki, public). Near-dedup
            (see ``near_dedup``) only actually runs for directories matching
            :data:`NEAR_DEDUP_DIR_MARKERS`; every directory still gets exact dedup.
        out_dir: Destination for cleaned shards.
        lang: ``"hindi"`` or ``"nepali"`` -- selects the profanity blocklist. Passed
            through to :func:`passes_filters`.
        seed_hashes: Exact-dedup hashes already present in a prior clean run, so
            re-running after collecting more raw data only processes genuinely new
            documents.
        contamination: Optional Phase 3 eval decontamination set (see
            :mod:`.decontam`). ``None`` or empty is a no-op -- there is nothing to
            decontaminate against until the Phase 3 synthetic reasoning dataset
            exists.
        near_dedup: Set False to disable near-dedup entirely (exact dedup only) even
            for directories that would otherwise qualify under
            :data:`NEAR_DEDUP_DIR_MARKERS`.

    ``out_dir`` is wiped first if it already has shards in it -- ``ShardWriter``
    resumes after the highest existing shard number rather than overwriting, which is
    right for an interrupted run but silently doubles the corpus if a finished clean
    is re-run on the same output directory (found the hard way: a re-run left the
    clean corpus with ~2x the correct document count, which then propagated into a
    doubled split downstream).
    """
    from .decontam import ContaminationSet, is_contaminated

    if out_dir.exists():
        shutil.rmtree(out_dir)
    stats = CleanStats()
    # Raw digest bytes, not the 64-char hex string -- at millions of documents the
    # hex form (64 bytes + ~50 bytes of Python str object overhead each) measurably
    # outgrows the 32-byte digest (+ ~33 bytes of bytes-object overhead), found the
    # hard way when this set's growth alone pushed a full-corpus clean run toward an
    # OOM kill on this machine's ~7GB RAM.
    seen_hashes: set[bytes] = {bytes.fromhex(h) for h in seed_hashes} if seed_hashes else set()
    contamination = contamination or ContaminationSet()
    dedup_filter = NearDedupFilter() if near_dedup else None
    writer = ShardWriter(out_dir)
    try:
        for input_dir in input_dirs:
            apply_near_dedup = near_dedup and _should_near_dedup(input_dir)
            for rec in iter_records(input_dir):
                stats.seen += 1
                text = normalize_text(rec.text)
                ok, reason = passes_filters(text, lang=lang)
                if not ok:
                    if reason == "short":
                        stats.dropped_short += 1
                    elif reason == "repetitive":
                        stats.dropped_repetitive += 1
                    elif reason == "profanity":
                        stats.dropped_profanity += 1
                    else:
                        stats.dropped_script += 1
                    continue
                if is_contaminated(text, contamination):
                    stats.dropped_contaminated += 1
                    continue
                digest = hashlib.sha256(text.encode("utf-8")).digest()
                if digest in seen_hashes:
                    stats.dropped_duplicate += 1
                    continue
                if apply_near_dedup and dedup_filter is not None and dedup_filter.is_near_duplicate(text):
                    stats.dropped_near_duplicate += 1
                    continue
                seen_hashes.add(digest)
                if apply_near_dedup and dedup_filter is not None:
                    dedup_filter.add(text)
                cleaned = Record.build(text=text, domain=rec.domain, method=rec.method, url=None)
                writer.write(cleaned)
                stats.kept += 1
                stats.output_bytes += cleaned.n_bytes
                stats.per_method[rec.method] = stats.per_method.get(rec.method, 0) + 1
    finally:
        writer.close_shard()
    return stats
