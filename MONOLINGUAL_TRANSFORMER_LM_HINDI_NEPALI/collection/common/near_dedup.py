"""MinHash/LSH near-duplicate detection, layered on top of clean.py's exact dedup.

Exact SHA-256 dedup (in clean.py) only catches byte-identical text. It misses the far
more common case in web-scraped news/public corpora: two documents that are almost the
same but differ by a byline, a dateline, a trailing "edited by" line, or a handful of
words -- wire-service reprints across outlets, or near-identical entries in a broad web
crawl like IndicCorp. MinHash + LSH banding catches these at a Jaccard-similarity
threshold without needing an all-pairs comparison (infeasible at millions of docs).

Memory design, both chosen after measuring the naive versions on this project's corpus
sizes on a RAM-constrained (~7GB) machine:

- ``LSHIndex`` stores each band as a *hashed* bucket key with a boolean presence
  marker, not the raw per-band shingle tuple plus a growing list of doc_ids per
  bucket. The naive version was estimated at ~8KB/document across all bands, which
  at ~1-2M documents would be several GB just for the index. This version is
  ~150-300 bytes/document (one int64 hash per band per document, no doc-id lists).
  The tradeoff: this index can only answer "has any doc already been seen in this
  bucket," not "which doc," so it flags near-duplicate *candidates* by whether a
  bucket was already occupied, then confirms with a direct Jaccard check against a
  small in-memory sample of already-kept shingle sets -- see ``NearDedupFilter``.
- MinHash signatures are computed with numpy, vectorized across all shingles and all
  permutations at once (broadcasted ``(a*shingle_hash + b) % p``, then a single
  ``min`` reduction along the shingle axis), not a nested Python loop over
  (permutation, shingle) pairs. The pure-Python version measured ~2-3ms/doc, which
  is tens of minutes of single-threaded work alone across a corpus this size; the
  vectorized version is roughly 50-100x faster.
- Hash arithmetic is masked to 32 bits (:data:`_HASH_MASK_32`), not the full 61-bit
  Mersenne-prime range MinHash theory technically allows, so that
  ``(a * shingle_hash)`` fits in a numpy ``uint64`` with no intermediate overflow
  before the modulo is applied. This trades a small amount of collision probability
  for correctness on numpy's fixed-width integer arithmetic.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field

import numpy as np

#: Word-shingle size for MinHash. 5 balances sensitivity (too small over-matches on
#: common short phrases) against recall (too large misses documents that share most
#: content but differ by a few inserted/removed words).
SHINGLE_SIZE = 5
#: Number of independent hash permutations in each MinHash signature.
NUM_PERMUTATIONS = 128
#: LSH banding: NUM_PERMUTATIONS must be exactly BANDS * ROWS_PER_BAND.
BANDS = 16
ROWS_PER_BAND = NUM_PERMUTATIONS // BANDS
#: A candidate pair colliding in >=1 band has an LSH-implied Jaccard threshold of
#: roughly (1/BANDS)^(1/ROWS_PER_BAND) ~= 0.72 at these settings -- pairs below this
#: are unlikely to collide in any band and won't be flagged (that's by design: this
#: is a near-duplicate filter, not a broader similarity/topic filter).
CONFIRM_JACCARD_THRESHOLD = 0.72

_HASH_MASK_32 = (1 << 32) - 1
#: A 61-bit Mersenne prime, standard for MinHash's universal hash family
#: ((a*x + b) mod p mod num_buckets); large enough relative to the 32-bit-masked
#: inputs that the modulo still spreads hash values well.
_MERSENNE_PRIME = (1 << 61) - 1


def word_shingles(text: str, k: int = SHINGLE_SIZE) -> set[str]:
    """Return the set of overlapping k-word shingles in ``text``."""
    words = text.split()
    if len(words) < k:
        return {" ".join(words)} if words else set()
    return {" ".join(words[i : i + k]) for i in range(len(words) - k + 1)}


class _Permutations:
    """Fixed, process-wide set of (a, b) coefficient pairs for the hash permutations.

    Generated once from a fixed seed so signatures are comparable across calls within
    a run (and reproducible across runs) -- MinHash only works if the same
    permutation family is used for every document being compared.
    """

    def __init__(self, num_permutations: int = NUM_PERMUTATIONS, seed: int = 1) -> None:
        rng = np.random.default_rng(seed)
        self.a = rng.integers(1, _HASH_MASK_32, size=num_permutations, dtype=np.uint64)
        self.b = rng.integers(0, _HASH_MASK_32, size=num_permutations, dtype=np.uint64)


_PERMS = _Permutations()


def _shingle_hashes(shingles: set[str]) -> np.ndarray:
    """Hash each shingle to a 32-bit-masked uint64, vectorizable in the signature step."""
    if not shingles:
        return np.array([], dtype=np.uint64)
    hashes = [
        int.from_bytes(hashlib.sha1(s.encode("utf-8")).digest()[:4], "big") & _HASH_MASK_32
        for s in shingles
    ]
    return np.array(hashes, dtype=np.uint64)


def minhash_signature(shingles: set[str]) -> np.ndarray:
    """Compute a MinHash signature (length :data:`NUM_PERMUTATIONS`) for a shingle set.

    Vectorized: broadcasts every permutation against every shingle hash in one numpy
    operation, rather than looping permutation-by-permutation or shingle-by-shingle in
    Python -- see the module docstring for why that matters at this corpus's scale.
    """
    hashes = _shingle_hashes(shingles)
    if hashes.size == 0:
        return np.zeros(NUM_PERMUTATIONS, dtype=np.uint64)
    # (num_permutations, num_shingles): each row is one permutation applied to every
    # shingle hash. uint64 arithmetic throughout stays within range because both `a`
    # and `hashes` are masked to 32 bits, so a*h fits in 64 bits before the modulo.
    permuted = (np.outer(_PERMS.a, hashes) + _PERMS.b[:, None]) % _MERSENNE_PRIME
    return permuted.min(axis=1).astype(np.uint64)


def jaccard_similarity(a: set[str], b: set[str]) -> float:
    """Exact Jaccard similarity between two shingle sets, for confirming an LSH candidate."""
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def _band_key(signature: np.ndarray, band_index: int) -> int:
    """Hash one band's slice of a signature down to a single bucket key."""
    start = band_index * ROWS_PER_BAND
    band_bytes = signature[start : start + ROWS_PER_BAND].tobytes()
    return int.from_bytes(hashlib.blake2b(band_bytes, digest_size=8).digest(), "big")


@dataclass
class LSHIndex:
    """Memory-light LSH bucket index: records which (band, bucket_key) pairs are occupied.

    Deliberately does *not* store which document occupies a bucket -- only a boolean
    "seen" marker per bucket, via a single global set of ``(band_index, bucket_key)``
    tuples. This means a lookup can only say "a same-band-bucket document already
    exists," not identify it -- which is enough to flag a *candidate*, but the actual
    similarity confirmation happens against a bounded recent-document sample (see
    :class:`NearDedupFilter`), not against the index itself.
    """

    seen_buckets: set[tuple[int, int]] = field(default_factory=set)

    def is_candidate(self, signature: np.ndarray) -> bool:
        """True if any band of ``signature`` collides with an already-indexed document."""
        for band in range(BANDS):
            if (band, _band_key(signature, band)) in self.seen_buckets:
                return True
        return False

    def add(self, signature: np.ndarray) -> None:
        for band in range(BANDS):
            self.seen_buckets.add((band, _band_key(signature, band)))


class NearDedupFilter:
    """Streaming near-duplicate filter: LSH candidate detection + a bounded confirm sample.

    Holds only a capped, FIFO-evicted sample of recent kept documents' shingle sets in
    memory for the Jaccard confirmation step -- not the full corpus's shingle sets,
    which would not fit in RAM at this project's document counts. This means a
    near-duplicate is only caught if its match is still in the sample window; near-dupes
    separated by more than ``sample_size`` kept documents in between can be missed.
    Acceptable here since near-duplicates in this project's sources (wire-service
    reprints, near-identical crawl entries) are typically collected close together in
    the raw stream, not scattered across the whole corpus.

    ``sample_size`` defaults small (4,000, not the originally-planned 20,000) because
    the dominant memory cost here is not the LSH index -- it's this sample, and its
    per-document cost was badly underestimated from a short synthetic test string
    before being measured against real corpus documents: real Hindi news articles'
    shingle sets average ~74KB each (not the ~4KB a 50-word toy example suggested), so
    a sample of 20,000 real documents was actually costing ~1.5GB, not the ~85MB
    originally estimated -- found only by watching RSS climb during an actual clean
    run, not by the earlier back-of-envelope math, which the corpus-size calculation
    in the module docstring's design notes should be read as a cautionary example of.
    """

    def __init__(self, sample_size: int = 4_000) -> None:
        self.index = LSHIndex()
        self.sample_size = sample_size
        self._sample: list[set[str]] = []
        self._sample_pos = 0

    def is_near_duplicate(self, text: str) -> bool:
        """Check ``text`` against the index/sample; does NOT record it (see :meth:`add`)."""
        shingles = word_shingles(text)
        if not shingles:
            return False
        signature = minhash_signature(shingles)
        if not self.index.is_candidate(signature):
            return False
        return any(jaccard_similarity(shingles, s) >= CONFIRM_JACCARD_THRESHOLD for s in self._sample)

    def add(self, text: str) -> None:
        """Record ``text`` as kept, for future candidate confirmation and indexing."""
        shingles = word_shingles(text)
        if not shingles:
            return
        signature = minhash_signature(shingles)
        self.index.add(signature)
        if len(self._sample) < self.sample_size:
            self._sample.append(shingles)
        else:
            self._sample[self._sample_pos] = shingles
            self._sample_pos = (self._sample_pos + 1) % self.sample_size
