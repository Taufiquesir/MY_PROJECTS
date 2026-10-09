"""One-off import of manually-collected shards gathered on a different machine.

Converts an external JSONL(.gz) export into this project's :class:`Record` schema
and routes each row into ``data/<lang>/manual_news/<domain>/`` by its own recorded
source domain, so it merges into the corpus exactly as if it had been crawled here.
Recomputes every derived field (n_chars, n_bytes, sha256) from the text itself rather
than trusting the source file's own copies, and re-derives doc_id from the text since
the export has no per-document URL to seed a URL-based id.

Usage:
    python -m collection.import_shard <lang> <path/to/export.jsonl.gz> [<path2> ...]
"""

from __future__ import annotations

import gzip
import json
import sys
from pathlib import Path

from .collect import data_root
from .common.record import METHOD_MANUAL, Record, ShardWriter


def _open(path: Path):
    return gzip.open(path, "rt", encoding="utf-8") if path.suffix == ".gz" else path.open("r", encoding="utf-8")


def import_shard(lang: str, path: Path) -> dict[str, int]:
    """Import one export file, returning a per-domain count of rows written."""
    writers: dict[str, ShardWriter] = {}
    counts: dict[str, int] = {}
    root = data_root() / lang / "manual_news"

    with _open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            text = row.get("text")
            if not text:
                continue
            domain = row.get("source") or row.get("domain") or "unknown"
            record = Record.build(text=text, domain=domain, method=METHOD_MANUAL)
            if domain not in writers:
                writers[domain] = ShardWriter(root / domain)
            writers[domain].write(record)
            counts[domain] = counts.get(domain, 0) + 1

    for w in writers.values():
        w.close_shard()
    return counts


def main() -> None:
    if len(sys.argv) < 3:
        print(__doc__)
        sys.exit(1)
    lang = sys.argv[1]
    for arg in sys.argv[2:]:
        path = Path(arg)
        print(f"[{lang}] importing {path} ...", flush=True)
        counts = import_shard(lang, path)
        for domain, n in sorted(counts.items()):
            print(f"    {domain}: {n:,} docs")


if __name__ == "__main__":
    main()
