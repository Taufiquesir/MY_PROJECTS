"""Deterministic document-level train/val/test split.

Splitting is done per document (not per line) using ``sha256(doc_id) % 100``, so the
assignment is reproducible without storing a random permutation and an article's
sentences never end up spread across splits.
"""

from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path

from .record import ShardWriter, iter_records

#: [0, 98) -> train, [98, 99) -> val, [99, 100) -> test.
TRAIN_CUTOFF = 98
VAL_CUTOFF = 99


def assign_split(doc_id: str) -> str:
    bucket = int(hashlib.sha256(doc_id.encode("utf-8")).hexdigest(), 16) % 100
    if bucket < TRAIN_CUTOFF:
        return "train"
    if bucket < VAL_CUTOFF:
        return "val"
    return "test"


def split_corpus(clean_dir: Path, out_root: Path, manifest_path: Path) -> dict:
    """Partition ``clean_dir`` into train/val/test shard directories under ``out_root``.

    Writes a JSON manifest (doc/byte counts per split) to ``manifest_path`` and
    returns the same dict.

    ``out_root`` is wiped first if it already has shards in it -- ``ShardWriter``
    resumes after the highest existing shard number rather than overwriting, which is
    the right behavior for an interrupted crawl but silently doubles the corpus if a
    finished split is re-run (found the hard way: a re-run left every split with ~2x
    the correct document count).
    """
    if out_root.exists():
        shutil.rmtree(out_root)
    writers = {name: ShardWriter(out_root / name) for name in ("train", "val", "test")}
    counts = {name: {"docs": 0, "bytes": 0} for name in writers}
    try:
        for rec in iter_records(clean_dir):
            split = assign_split(rec.doc_id)
            writers[split].write(rec)
            counts[split]["docs"] += 1
            counts[split]["bytes"] += rec.n_bytes
    finally:
        for w in writers.values():
            w.close_shard()

    manifest = {"splits": counts}
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest
