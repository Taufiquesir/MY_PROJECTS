"""Document record schema and shard I/O shared by every collector.

Every piece of text gathered for this project -- whether pulled by my own crawler or
streamed from a public corpus -- gets wrapped as one :class:`Record` and appended to a
JSONL shard. Storing the collection method on the record itself means the manual-vs-
downloaded token table the assignment asks for is just a groupby over this one field,
instead of something I have to track separately.
"""

from __future__ import annotations

import gzip
import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Iterator

#: Number of records held in a shard before it is closed and gzipped.
SHARD_CAPACITY = 20_000

#: collection_method value for text I fetched myself (news crawl or Wikipedia dump).
METHOD_MANUAL = "manual"
#: collection_method value for text pulled from a public corpus (Sangraha).
METHOD_PUBLIC = "public"


@dataclass(frozen=True)
class Record:
    """One document plus the provenance fields Phase 1 reporting needs.

    Attributes:
        doc_id: First 16 hex chars of sha256(url or text) -- stable across re-runs so
            collecting twice does not create duplicate ids.
        text: Raw document text at collection time (NFC-normalized later by cleaning).
        domain: Site hostname for crawled text, or corpus name for public text.
        method: :data:`METHOD_MANUAL` or :data:`METHOD_PUBLIC`.
        n_chars: Length in Unicode codepoints.
        n_bytes: Length in UTF-8 bytes.
        sha256: Full digest of ``text``, used for exact-match dedup.
    """

    doc_id: str
    text: str
    domain: str
    method: str
    n_chars: int
    n_bytes: int
    sha256: str

    @staticmethod
    def build(text: str, domain: str, method: str, url: str | None = None) -> "Record":
        """Construct a Record, deriving every provenance field from ``text``.

        Args:
            text: Document body.
            domain: Hostname (crawled) or corpus name (public).
            method: :data:`METHOD_MANUAL` or :data:`METHOD_PUBLIC`.
            url: Source URL, when there is one. Used to seed ``doc_id`` so refetching
                the same URL always yields the same id; falls back to hashing the text
                itself for sources with no natural URL (e.g. Sangraha rows).
        """
        digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
        id_seed = url if url is not None else text
        return Record(
            doc_id=hashlib.sha256(id_seed.encode("utf-8")).hexdigest()[:16],
            text=text,
            domain=domain,
            method=method,
            n_chars=len(text),
            n_bytes=len(text.encode("utf-8")),
            sha256=digest,
        )

    def dumps(self) -> str:
        """Serialize to a single JSON line (no trailing newline)."""
        return json.dumps(asdict(self), ensure_ascii=False)


class ShardWriter:
    """Append-only JSONL shard writer that gzips shards once they fill up.

    Picks up after the highest-numbered shard already on disk, so calling this again
    after an interrupted run appends new shards rather than overwriting old ones.
    """

    def __init__(self, out_dir: Path, capacity: int = SHARD_CAPACITY) -> None:
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.capacity = capacity
        self._shard_no = self._resume_index()
        self._fh = None
        self._in_shard = 0
        self.total_written = 0

    def _resume_index(self) -> int:
        existing = [
            int(p.name.split(".")[0].split("-")[1])
            for p in self.out_dir.glob("shard-*.jsonl*")
        ]
        return max(existing) + 1 if existing else 0

    def _open_shard(self) -> None:
        path = self.out_dir / f"shard-{self._shard_no:05d}.jsonl"
        self._fh = path.open("a", encoding="utf-8")
        self._in_shard = 0

    def write(self, record: Record) -> None:
        """Append one record, sealing the current shard once it hits capacity."""
        if self._fh is None:
            self._open_shard()
        self._fh.write(record.dumps() + "\n")
        self._in_shard += 1
        self.total_written += 1
        if self._in_shard >= self.capacity:
            self.close_shard()

    def flush(self) -> None:
        """Flush the OS write buffer for the open shard, without sealing it.

        Called after every fetch during a crawl so a mid-run crash loses at most the
        URL currently in flight, never a document already reported as done.
        """
        if self._fh is not None:
            self._fh.flush()

    def close_shard(self) -> None:
        """Close and gzip the current shard, then advance to the next index."""
        if self._fh is None:
            return
        self._fh.flush()
        path = Path(self._fh.name)
        self._fh.close()
        self._fh = None
        if path.stat().st_size > 0:
            with path.open("rb") as src, gzip.open(f"{path}.gz", "wb") as dst:
                dst.writelines(src)
        path.unlink(missing_ok=True)
        self._shard_no += 1
        self._in_shard = 0

    def __enter__(self) -> "ShardWriter":
        return self

    def __exit__(self, *exc_info) -> None:
        self.close_shard()


def iter_records(shard_dir: Path) -> Iterator[Record]:
    """Yield every :class:`Record` under ``shard_dir``, across plain and gzipped shards.

    Searches recursively (``**/shard-*.jsonl*``) so a directory holding one
    subdirectory of shards per domain -- as the concurrent news crawler writes, to
    give each domain its own :class:`ShardWriter` shard-index namespace -- is read the
    same as a flat directory of shards.

    A shard whose last line was cut off by a crash is not treated as an error -- the
    broken line is skipped, since :class:`ShardWriter` only ever appends complete lines
    before the interruption.

    Some Nepali split shards from an earlier export predate the current 7-field
    schema and only have ``doc_id``/``text`` -- rather than silently dropping every
    line in those shards (the old behavior: any TypeError from a missing field was
    swallowed), missing provenance fields are backfilled from ``text`` here so
    training can still read the document. The backfilled ``domain``/``method`` are
    placeholders ("unknown"/"unknown"), not real provenance -- fine for training,
    which only reads ``.text``, but do not trust them for Phase 1 manual-vs-public
    reporting without re-deriving from the original source.
    """
    shard_dir = Path(shard_dir)
    if not shard_dir.exists():
        return
    for path in sorted(shard_dir.glob("**/shard-*.jsonl*")):
        opener = gzip.open if path.suffix == ".gz" else open
        with opener(path, "rt", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if "text" not in row or "doc_id" not in row:
                    continue
                row.setdefault("domain", "unknown")
                row.setdefault("method", "unknown")
                text = row["text"]
                row.setdefault("n_chars", len(text))
                row.setdefault("n_bytes", len(text.encode("utf-8")))
                row.setdefault("sha256", hashlib.sha256(text.encode("utf-8")).hexdigest())
                try:
                    yield Record(**row)
                except TypeError:
                    continue


def write_records(shard_dir: Path, records: Iterable[Record]) -> int:
    """Write ``records`` to fresh shards under ``shard_dir``; return count written."""
    with ShardWriter(shard_dir) as writer:
        for record in records:
            writer.write(record)
        return writer.total_written
