"""Downloads Hindi/Nepali text from AI4Bharat IndicCorpV2 as the public/downloaded corpus.

Replaces Sangraha for both languages' downloaded slice: IndicCorpV2 ships plain UTF-8
``.txt`` files (one document/line, no Parquet/schema handling needed), which is
simpler to work with and avoids the CDN read-timeout issues hit repeatedly downloading
Sangraha's Parquet shards on this connection.

Per-language file layout differs on the HuggingFace side: Nepali is one file
(``data/ne.txt``, ~14.77 GB); Hindi is split into three roughly-26.66 GB files
(``data/hi-1.txt``, ``hi-2.txt``, ``hi-3.txt``) -- only ``hi-1.txt`` is used here since
one file alone is far larger than any per-language token target in this project.
Either way this only downloads a byte-range prefix of the file via HTTP Range rather
than the whole thing, sized from each language's own chars-per-token planning rate,
then trims to the last complete line so no row is cut mid-document.

Downloaded in parallel: the target byte range is split into fixed-size pieces (default
64 MB) fetched by a pool of worker threads, each via its own ``Range: bytes=a-b``
request, written directly to its own offset in the pre-sized output file (no merge
step needed). This is a straightforward win on a connection whose bottleneck is
per-connection throughput rather than total bandwidth -- confirmed by hand here: a
single stream held a steady ~5-6 MB/s with headroom unused elsewhere, and 8 parallel
streams reached ~50 MB/s on the same connection.

Output: a plain-text file (one document per line) under ``data/<lang>/indiccorp/``,
matching the "raw scraped text" storage convention used by the manual-collection
stages -- these are converted into the project's Record schema and merged into the
corpus at the ``clean`` stage, same as any other source.

Usage::

    python -m collection.download_indiccorp --lang nepali --tokens 450000000
    python -m collection.download_indiccorp --lang hindi  --tokens 500000000
"""

from __future__ import annotations

import argparse
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import requests

RESOLVE_URL = "https://huggingface.co/datasets/ai4bharat/IndicCorpV2/resolve/main/data/{filename}"
#: Per-language source filename (without the .txt extension) on the HF repo.
LANG_FILE = {"nepali": "ne", "hindi": "hi-1"}
#: Per-language output filename under data/<lang>/indiccorp/ -- kept simple (just the
#: language code) regardless of which upstream file/shard it was sourced from.
LANG_OUTPUT_NAME = {"nepali": "ne", "hindi": "hi"}

#: Pre-tokenizer planning rate, per language -- Hindi and Nepali were each measured/
#: specified separately (4.3 for Hindi, 4.0 for Nepali) rather than sharing one
#: constant, since the two corpora and their tokenizers are fully independent per the
#: assignment's no-shared-vocabulary requirement.
CHARS_PER_TOKEN = {"nepali": 4.0, "hindi": 4.3}
#: Devanagari text in UTF-8 averages ~3 bytes/char (most codepoints fall in the
#: 2-3-byte range; combining marks add a little more) -- a planning estimate only,
#: replaced by the tokenizer's own measured rate once one is trained. Same for both
#: languages since both are Devanagari.
BYTES_PER_CHAR = 3.0

CHUNK_BYTES = 64 * 1024 * 1024
WORKERS = 8


def _download_range(url: str, start: int, end: int, retries: int = 6) -> bytes:
    """Fetch ``bytes=start-end`` (inclusive) of ``url``, retrying transient failures."""
    for attempt in range(1, retries + 1):
        try:
            resp = requests.get(url, headers={"Range": f"bytes={start}-{end}"}, timeout=(15, 90))
            if resp.status_code not in (200, 206):
                raise requests.RequestException(f"status {resp.status_code}")
            return resp.content
        except requests.RequestException as exc:
            print(f"[indiccorp] range {start}-{end} attempt {attempt}/{retries} failed: {type(exc).__name__}", flush=True)
            time.sleep(min(3 * attempt, 20))
    raise RuntimeError(f"giving up on range {start}-{end} after {retries} attempts")


def _load_completed_chunks(manifest_path: Path) -> set[int]:
    if not manifest_path.exists():
        return set()
    return {int(line) for line in manifest_path.read_text("utf-8").splitlines() if line.strip()}


def _mark_chunk_complete(manifest_path: Path, start: int) -> None:
    with manifest_path.open("a", encoding="utf-8") as f:
        f.write(f"{start}\n")


def _download_prefix_parallel(url: str, dest: Path, target_bytes: int, workers: int = WORKERS, chunk_bytes: int = CHUNK_BYTES) -> None:
    """Download the first ``target_bytes`` of ``url`` to ``dest`` using parallel Range requests.

    The output file is pre-sized to ``target_bytes`` up front (via ``truncate``) so
    each worker can seek-and-write its own chunk directly with no merge step and no
    requirement that chunks complete in order -- but that means the file's *apparent*
    size hits ``target_bytes`` immediately, before any chunk has actually been
    written, so file size cannot be used to detect what has really been downloaded.
    A sidecar ``<dest>.chunks`` manifest instead records each chunk's start offset
    once it is confirmed written, and resuming reads that manifest rather than
    trusting ``dest.stat().st_size`` -- found the hard way: an earlier version used
    file size for resume, which after any interruption saw the pre-sized (but mostly
    still-sparse/zero) file as "already complete" and skipped re-downloading the
    chunks that were never actually written, silently leaving zero-byte holes in the
    middle of the corpus text.
    """
    dest.parent.mkdir(parents=True, exist_ok=True)
    manifest_path = dest.with_suffix(dest.suffix + ".chunks")
    completed_starts = _load_completed_chunks(manifest_path)

    # Pre-size the file so workers can write to arbitrary offsets independently.
    # Safe to always (re-)truncate to the same target_bytes even on a resume: growing
    # or matching size never touches bytes already written by a completed chunk.
    with dest.open("r+b" if dest.exists() else "wb") as f:
        f.truncate(target_bytes)

    all_chunks = [(start, min(start + chunk_bytes, target_bytes) - 1) for start in range(0, target_bytes, chunk_bytes)]
    chunks = [(start, end) for start, end in all_chunks if start not in completed_starts]
    if not chunks:
        print(f"[indiccorp] all {len(all_chunks)} chunks already downloaded, target met", flush=True)
        return

    total_mb = target_bytes / 1e6
    already_mb = sum(min(s + chunk_bytes, target_bytes) - s for s in completed_starts) / 1e6
    progress = {"done_mb": already_mb}
    lock = threading.Lock()
    print(
        f"[indiccorp] {len(completed_starts)}/{len(all_chunks)} chunks already done, "
        f"downloading {len(chunks)} remaining with {workers} workers",
        flush=True,
    )

    def fetch_and_write(start: int, end: int) -> int:
        content = _download_range(url, start, end)
        # Separate fd per write -- concurrent threads must not share one file
        # object's cursor, since pwrite-style positional writes are what makes
        # out-of-order completion safe here.
        fd = os.open(dest, os.O_WRONLY)
        try:
            os.pwrite(fd, content, start)
        finally:
            os.close(fd)
        _mark_chunk_complete(manifest_path, start)
        return len(content)

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(fetch_and_write, start, end): (start, end) for start, end in chunks}
        for future in as_completed(futures):
            n = future.result()
            with lock:
                progress["done_mb"] += n / 1e6
                done_mb = progress["done_mb"]
            print(f"[indiccorp] {done_mb:.0f} / {total_mb:.0f} MB downloaded so far", flush=True)


def _trim_to_complete_lines(path: Path, tail_scan_bytes: int = 8 * 1024 * 1024) -> None:
    """Drop any trailing partial line left by a byte-range cutoff mid-document.

    Only reads/rewrites the last ``tail_scan_bytes`` of the file, not the whole
    thing -- these files run several GB, and loading one fully into memory (an
    earlier version of this function did exactly that, via ``f.read()``) risks an
    OOM kill on a machine this tight on RAM. Truncating in place via ``os.truncate``
    likewise never requires holding the file's contents in memory.
    """
    with path.open("rb") as f:
        f.seek(0, 2)
        size = f.tell()
        if size == 0:
            return
        f.seek(max(0, size - 1))
        last_byte = f.read(1)
        if last_byte == b"\n":
            return
        scan_start = max(0, size - tail_scan_bytes)
        f.seek(scan_start)
        tail = f.read()
    last_newline = tail.rfind(b"\n")
    if last_newline == -1:
        raise RuntimeError(
            f"no newline found in the last {tail_scan_bytes} bytes of {path} -- "
            "increase tail_scan_bytes or investigate the file"
        )
    new_size = scan_start + last_newline + 1
    with path.open("r+b") as f:
        f.truncate(new_size)


def download_indiccorp(lang: str, target_tokens: int, out_dir: Path) -> Path:
    """Download a byte-range prefix of IndicCorpV2's language file, sized for ``target_tokens``.

    Returns the path to the downloaded ``.txt`` file under ``out_dir``.
    """
    source_filename = LANG_FILE[lang]
    output_name = LANG_OUTPUT_NAME[lang]
    chars_per_token = CHARS_PER_TOKEN[lang]
    target_bytes = int(target_tokens * chars_per_token * BYTES_PER_CHAR)
    url = RESOLVE_URL.format(filename=f"{source_filename}.txt")
    dest = out_dir / f"{output_name}.txt"

    print(
        f"[indiccorp] {lang}: target ~{target_tokens:,} tokens at {chars_per_token} chars/token "
        f"-> {target_bytes/1e9:.2f} GB of {source_filename}.txt",
        flush=True,
    )
    _download_prefix_parallel(url, dest, target_bytes)
    _trim_to_complete_lines(dest)
    final_size = dest.stat().st_size
    print(f"[indiccorp] {lang}: done: {dest} ({final_size/1e9:.2f} GB)", flush=True)
    return dest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lang", required=True, choices=tuple(LANG_FILE))
    parser.add_argument("--tokens", type=int, required=True)
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    args = parser.parse_args()

    out_dir = args.data_root / args.lang / "indiccorp"
    download_indiccorp(args.lang, args.tokens, out_dir)


if __name__ == "__main__":
    main()
