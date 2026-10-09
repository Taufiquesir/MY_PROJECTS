"""Streams the downloaded (public) slice of each corpus from AI4Bharat Sangraha.

Sangraha (https://huggingface.co/datasets/ai4bharat/sangraha) ships its ``verified``
split as one Parquet file per shard (~200-400MB each). ``datasets.load_dataset(...,
streaming=True)`` was tried first and abandoned: on this machine's connection (a
throttled ~4 MB/s to HF's CDN), its lazy fsspec row-group reads stalled indefinitely
on the first row rather than degrading to a slow-but-finishing download. Downloading
each shard file-at-a-time with plain ``requests`` (resumable via HTTP Range, retried
on transient failures) and then reading it with ``pyarrow`` is slower to start but
actually completes, and each shard is deleted right after being folded into
``public/`` so raw Parquet never accumulates on disk.

Only the ``verified`` split is used, never ``synthetic`` (that subset is machine-
translated/back-transliterated text, not organic monolingual text, and mixing it in
would undermine the "own monolingual corpus" requirement).
"""

from __future__ import annotations

import time
from pathlib import Path

import pyarrow.parquet as pq
import requests
from huggingface_hub import HfApi

from .record import METHOD_PUBLIC, Record, ShardWriter, iter_records

RESOLVE_TEMPLATE = "https://huggingface.co/datasets/ai4bharat/sangraha/resolve/main/{path}"


def _list_shard_paths(sangraha_dir: str, retries: int = 4) -> list[str]:
    """Return every Parquet shard path under ``sangraha_dir`` (e.g. ``verified/hin``).

    Retried on any transport error (DNS resolution failure, connection reset) --
    this call is a single request at the very start of the run, so an unretried
    transient network blip here used to crash the whole ingest before it downloaded
    a single byte.
    """
    api = HfApi()
    for attempt in range(retries):
        try:
            files = api.list_repo_files("ai4bharat/sangraha", repo_type="dataset")
            break
        except requests.RequestException as exc:
            if attempt == retries - 1:
                raise
            print(f"[sangraha] list_repo_files attempt {attempt + 1}/{retries} failed ({exc}), retrying", flush=True)
            time.sleep(3 * (attempt + 1))
    prefix = sangraha_dir.rstrip("/") + "/"
    return sorted(f for f in files if f.startswith(prefix) and f.endswith(".parquet"))


def _download_shard(remote_path: str, dest: Path, retries: int = 8, progress_every_mb: int = 50) -> bool:
    """Download one shard to ``dest``, resuming a partial file via HTTP Range.

    Prints a heartbeat every ``progress_every_mb`` MB of this attempt's download --
    a single shard can be several hundred MB and takes minutes on a throttled
    connection, during which the caller's own log otherwise goes silent with no way
    to distinguish "still downloading" from "hung."

    ``retries`` defaults higher than a typical transient-failure retry count (8, not
    3) because HuggingFace's Xet CDN backend (``us.aws.cdn.hf.co``) has been observed
    read-timing-out repeatedly on this connection -- not a single blip but a run of
    several drops in a row -- while still making forward progress overall each time
    it does connect. Each retry resumes via Range from wherever the previous attempt
    left off (``existing`` below), so raising the retry count costs wall-clock time
    on a flaky connection but never repeats already-downloaded bytes.

    Crucially the budget is only consumed by attempts that transferred *nothing*.
    On a connection dropping every ~15MB, a 378MB shard needs ~25 successful
    resumes to finish; charging those against a fixed retry count meant the shard
    was abandoned while it was still steadily progressing, and since a partial file
    is discarded on give-up, such a shard could never complete no matter how long
    it ran. An attempt that moved bytes forward resets the budget instead.

    Returns False (without raising) if the budget of no-progress attempts is
    exhausted, so the caller can skip a persistently broken shard instead of
    aborting the whole ingest run.
    """
    url = RESOLVE_TEMPLATE.format(path=remote_path)
    attempt = 0
    stalled = 0
    while stalled < retries:
        attempt += 1
        try:
            existing = dest.stat().st_size if dest.exists() else 0
            headers = {"Range": f"bytes={existing}-"} if existing else {}
            downloaded_this_attempt = 0
            next_report_at = progress_every_mb * 1024 * 1024
            with requests.get(url, headers=headers, stream=True, timeout=(15, 90)) as resp:
                if resp.status_code not in (200, 206):
                    raise requests.RequestException(f"status {resp.status_code}")
                mode = "ab" if resp.status_code == 206 else "wb"
                with dest.open(mode) as f:
                    for chunk in resp.iter_content(chunk_size=1024 * 1024):
                        f.write(chunk)
                        downloaded_this_attempt += len(chunk)
                        if downloaded_this_attempt >= next_report_at:
                            total_mb = (existing + downloaded_this_attempt) / 1e6
                            print(f"[sangraha] {remote_path}: {total_mb:.0f} MB downloaded so far", flush=True)
                            next_report_at += progress_every_mb * 1024 * 1024
            return True
        except requests.RequestException as exc:
            if downloaded_this_attempt > 0:
                # Made real forward progress before dropping -- Range will resume from
                # here, so this is not a stall and must not consume the retry budget.
                stalled = 0
                note = f"+{downloaded_this_attempt / 1e6:.0f} MB before drop"
            else:
                stalled += 1
                note = f"no progress, {stalled}/{retries}"
            print(f"[sangraha] {remote_path}: attempt {attempt} failed ({note}): {type(exc).__name__}", flush=True)
            time.sleep(min(3 * (stalled + 1), 20))
    return False


def _ingest_parquet_file(
    local_path: Path,
    sangraha_dir: str,
    writer: ShardWriter,
    already_plus_collected: "list[int]",
    target_bytes: int,
    lang: str,
) -> bool:
    """Read one local Parquet file's ``text`` column into ``writer`` as Records.

    ``already_plus_collected`` is a 1-element list used as a mutable int cell (so
    this helper can update the caller's running total) -- shared between the
    network-download path and the local-file path so both stop at ``target_bytes``
    and report progress the same way.

    Returns True if the file was read to completion (so the caller can mark it
    "done" in the shard manifest), False if it stopped early because the target was
    reached or the file could not be parsed at all.
    """
    docs = 0
    try:
        pf = pq.ParquetFile(local_path)
        for batch in pf.iter_batches(batch_size=2000, columns=["text"]):
            for text in batch.column("text").to_pylist():
                if not text:
                    continue
                record = Record.build(text=text, domain=f"sangraha/{sangraha_dir}", method=METHOD_PUBLIC)
                writer.write(record)
                already_plus_collected[0] += record.n_bytes
                docs += 1
                if docs % 5000 == 0:
                    writer.flush()
                    print(f"[{lang}] public: {docs:,} docs, {already_plus_collected[0]/1e6:.1f} MB total", flush=True)
            if already_plus_collected[0] >= target_bytes:
                return False
        return True
    except Exception as exc:  # malformed/partial shard -- skip, don't abort the run
        print(f"[{lang}] public: could not read {local_path} ({exc}), skipping", flush=True)
        return False


def ingest_local_shards(lang: str, sangraha_dir: str, out_dir: Path, target_bytes: int, local_dir: Path) -> int:
    """Fold already-downloaded Sangraha Parquet shards from ``local_dir`` into ``out_dir``.

    For when the network path (:func:`ingest_sangraha`) is too unreliable to finish a
    shard download on this machine's connection -- shards fetched by hand elsewhere
    (browser, huggingface-cli, a different machine) and copied into ``local_dir`` are
    read directly with no download step. Shares the same ``sangraha_completed_shards.txt``
    manifest as the network path, keyed by the shard's remote path
    (``<sangraha_dir>/<filename>``), so a shard ingested this way is not re-downloaded
    later if the network path is also run, and vice versa.

    Args:
        lang: Project language name, used only for log messages.
        sangraha_dir: Sangraha's own subset path, e.g. ``"verified/hin"`` -- used to
            build the manifest key and the ``domain`` field on each Record, not to
            fetch anything.
        out_dir: Shard directory to append cleaned :class:`Record` rows to.
        target_bytes: Stop once this many UTF-8 text bytes have been collected in
            total (across this and any prior run into the same directory).
        local_dir: Directory containing already-downloaded ``*.parquet`` files.

    Returns:
        Text bytes collected this run.
    """
    already = sum(r.n_bytes for r in iter_records(out_dir))
    if already >= target_bytes:
        print(f"[{lang}] public: already at {already/1e6:.1f} MB, target met", flush=True)
        return 0

    manifest_path = out_dir.parent / "sangraha_completed_shards.txt"
    completed = _load_completed_shards(manifest_path)

    local_files = sorted(Path(local_dir).glob("*.parquet"))
    print(f"[{lang}] public: {len(local_files)} local shard(s) found in {local_dir}", flush=True)

    writer = ShardWriter(out_dir)
    running_total = [already]
    try:
        for local_path in local_files:
            remote_key = f"{sangraha_dir.rstrip('/')}/{local_path.name}"
            if remote_key in completed:
                print(f"[{lang}] public: {local_path.name} already ingested, skipping", flush=True)
                continue
            if running_total[0] >= target_bytes:
                break
            print(f"[{lang}] public: ingesting local file {local_path}", flush=True)
            finished = _ingest_parquet_file(local_path, sangraha_dir, writer, running_total, target_bytes, lang)
            if finished:
                _mark_shard_complete(manifest_path, remote_key)
    finally:
        writer.close_shard()
    collected = running_total[0] - already
    print(f"[{lang}] public: stopped at {collected/1e6:.1f} MB collected this run ({running_total[0]/1e6:.1f} MB total)", flush=True)
    return collected


def _load_completed_shards(manifest_path: Path) -> set[str]:
    if not manifest_path.exists():
        return set()
    return {line.strip() for line in manifest_path.read_text("utf-8").splitlines() if line.strip()}


def _mark_shard_complete(manifest_path: Path, remote_path: str) -> None:
    with manifest_path.open("a", encoding="utf-8") as f:
        f.write(remote_path + "\n")


def ingest_sangraha(lang: str, sangraha_dir: str, out_dir: Path, target_bytes: int, scratch_dir: Path | None = None) -> int:
    """Download and fold Sangraha's ``verified/<code>`` shards into ``out_dir``.

    Args:
        lang: Project language name, used only for log messages.
        sangraha_dir: Sangraha's own subset path, e.g. ``"verified/hin"``.
        out_dir: Shard directory to append cleaned :class:`Record` rows to.
        target_bytes: Stop once this many UTF-8 text bytes have been collected in
            total (across this and any prior run into the same directory).
        scratch_dir: Where to download raw Parquet shards before deleting them.
            Defaults to a ``_scratch_sangraha`` sibling of ``out_dir``.

    Returns:
        Text bytes collected this run.

    A ``completed_shards.txt`` manifest next to ``out_dir`` records which remote
    Parquet shards have already been fully folded in, and is checked before
    (re-)downloading a shard. Without this, restarting the ingest after any
    interruption -- and every interruption seen in practice so far has been a
    transient network/OOM issue, not a genuine "done" stop -- always started over
    from the first shard, since the only progress signal (``already`` = total bytes
    on disk) says *how much* was collected but not *which shards* produced it. That
    silently re-downloaded and re-wrote already-ingested documents as duplicate
    records on every restart -- caught by inspecting doc_id uniqueness in
    ``public/`` after several restarts (349,526 records, only 343,499 unique
    ids). Sangraha rows have no natural id to dedupe on the way in (Record.build
    falls back to hashing the text itself when no url is given), so shard-level
    resume is the fix, not per-document dedup here -- exact-dedup already happens
    later in clean_corpus for any duplicates that still slip through.
    """
    already = sum(r.n_bytes for r in iter_records(out_dir))
    if already >= target_bytes:
        print(f"[{lang}] public: already at {already/1e6:.1f} MB, target met", flush=True)
        return 0

    scratch = Path(scratch_dir) if scratch_dir else out_dir.parent / "_scratch_sangraha"
    scratch.mkdir(parents=True, exist_ok=True)
    manifest_path = out_dir.parent / "sangraha_completed_shards.txt"
    completed = _load_completed_shards(manifest_path)

    shard_paths = _list_shard_paths(sangraha_dir)
    todo = [p for p in shard_paths if p not in completed]
    print(
        f"[{lang}] public: {len(shard_paths)} shards available under {sangraha_dir}, "
        f"{len(completed)} already ingested, {len(todo)} remaining",
        flush=True,
    )

    writer = ShardWriter(out_dir)
    collected = 0
    docs = 0
    try:
        for remote_path in todo:
            if already + collected >= target_bytes:
                break
            local_path = scratch / Path(remote_path).name
            print(f"[{lang}] public: downloading {remote_path}", flush=True)
            ok = _download_shard(remote_path, local_path)
            if not ok:
                print(f"[{lang}] public: giving up on {remote_path} after retries", flush=True)
                local_path.unlink(missing_ok=True)
                continue

            try:
                pf = pq.ParquetFile(local_path)
                for batch in pf.iter_batches(batch_size=2000, columns=["text"]):
                    for text in batch.column("text").to_pylist():
                        if not text:
                            continue
                        record = Record.build(text=text, domain=f"sangraha/{sangraha_dir}", method=METHOD_PUBLIC)
                        writer.write(record)
                        collected += record.n_bytes
                        docs += 1
                        if docs % 5000 == 0:
                            writer.flush()
                            print(f"[{lang}] public: {docs:,} docs, {collected/1e6:.1f} MB this run", flush=True)
                    if already + collected >= target_bytes:
                        break
                else:
                    # Only reached if the batch loop completed without an early
                    # break, i.e. this shard was read in full -- do not mark a
                    # shard "complete" if we stopped partway through it on
                    # hitting target_bytes, since a later run with a higher
                    # target needs to resume it, not skip it as already done.
                    _mark_shard_complete(manifest_path, remote_path)
            except Exception as exc:  # malformed/partial shard -- skip, don't abort the run
                print(f"[{lang}] public: could not read {remote_path} ({exc}), skipping", flush=True)
            finally:
                local_path.unlink(missing_ok=True)
    finally:
        writer.close_shard()
    print(f"[{lang}] public: stopped at {docs:,} docs, {collected/1e6:.1f} MB this run", flush=True)
    return collected
