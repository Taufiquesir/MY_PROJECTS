"""Converts a downloaded IndicCorpV2 ``.txt`` file into the project's Record schema.

IndicCorpV2's per-language files (``data/<code>.txt``) are one document per line,
plain UTF-8. This reads that file and writes it into ``data/<lang>/public/`` as
gzipped JSONL shards, same as :mod:`collection.common.public_corpus` does for
Sangraha -- both land in the same ``public`` directory and are read identically by
the ``clean`` stage, since ``method="public"`` is what the assignment's
manual-vs-downloaded split actually keys on, not which corpus a document came from.

Usage::

    python -m collection.import_indiccorp --lang nepali --path data/nepali/indiccorp/ne.txt
"""

from __future__ import annotations

import argparse
from pathlib import Path

from collection.common.record import METHOD_PUBLIC, Record, ShardWriter


def import_indiccorp_file(lang: str, txt_path: Path, out_dir: Path) -> int:
    """Import one IndicCorpV2 ``.txt`` file (one document per line) into ``out_dir``.

    Returns the number of documents written.
    """
    writer = ShardWriter(out_dir)
    n = 0
    try:
        with txt_path.open("r", encoding="utf-8") as f:
            for line in f:
                text = line.strip()
                if not text:
                    continue
                record = Record.build(text=text, domain=f"indiccorp/{lang}", method=METHOD_PUBLIC)
                writer.write(record)
                n += 1
                if n % 50_000 == 0:
                    writer.flush()
                    print(f"[indiccorp] {lang}: {n:,} docs imported", flush=True)
    finally:
        writer.close_shard()
    print(f"[indiccorp] {lang}: {n:,} docs imported total from {txt_path}", flush=True)
    return n


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lang", required=True)
    parser.add_argument("--path", type=Path, required=True, help="Path to the downloaded .txt file")
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    args = parser.parse_args()

    out_dir = args.data_root / args.lang / "public"
    import_indiccorp_file(args.lang, args.path, out_dir)


if __name__ == "__main__":
    main()
