"""Corpus statistics: per-source/per-method document and byte breakdown.

Kept deliberately simple -- one pass over the cleaned corpus, grouping by the
provenance fields already stored on every :class:`~.record.Record`. This is what
makes the assignment's required manual-vs-downloaded token table reproducible
directly from the data rather than tracked by hand during collection.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

from .record import iter_records


@dataclass
class CorpusStats:
    docs: int = 0
    chars: int = 0
    bytes: int = 0
    per_domain_docs: dict[str, int] = field(default_factory=dict)
    per_domain_bytes: dict[str, int] = field(default_factory=dict)
    per_method_docs: dict[str, int] = field(default_factory=dict)
    per_method_bytes: dict[str, int] = field(default_factory=dict)

    def as_dict(self) -> dict:
        return {
            "docs": self.docs,
            "chars": self.chars,
            "bytes": self.bytes,
            "per_domain_docs": self.per_domain_docs,
            "per_domain_bytes": self.per_domain_bytes,
            "per_method_docs": self.per_method_docs,
            "per_method_bytes": self.per_method_bytes,
        }


def compute_corpus_stats(clean_dir: Path) -> CorpusStats:
    """Compute document/byte counts, grouped by domain and by collection method."""
    stats = CorpusStats()
    for rec in iter_records(clean_dir):
        stats.docs += 1
        stats.chars += rec.n_chars
        stats.bytes += rec.n_bytes
        stats.per_domain_docs[rec.domain] = stats.per_domain_docs.get(rec.domain, 0) + 1
        stats.per_domain_bytes[rec.domain] = stats.per_domain_bytes.get(rec.domain, 0) + rec.n_bytes
        stats.per_method_docs[rec.method] = stats.per_method_docs.get(rec.method, 0) + 1
        stats.per_method_bytes[rec.method] = stats.per_method_bytes.get(rec.method, 0) + rec.n_bytes
    return stats


def write_json(payload: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def corpus_independence(clean_dir_a: Path, clean_dir_b: Path) -> dict:
    """Confirm two languages' cleaned corpora share zero documents, by text hash."""
    hashes_a = {rec.sha256 for rec in iter_records(clean_dir_a)}
    hashes_b = {rec.sha256 for rec in iter_records(clean_dir_b)}
    shared = hashes_a & hashes_b
    return {
        "docs_a": len(hashes_a),
        "docs_b": len(hashes_b),
        "shared_docs": len(shared),
        "independent": len(shared) == 0,
    }
