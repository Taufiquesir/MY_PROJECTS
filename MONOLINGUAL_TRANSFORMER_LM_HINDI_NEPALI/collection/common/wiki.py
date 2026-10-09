"""Wikipedia ingest: a second, non-news manual-collection source per language.

News crawling alone tends to over-represent one house style and one genre (current
affairs). Walking each language's Wikipedia via the MediaWiki API adds encyclopedic,
multi-topic text collected by hand (not from a static dump), broadening the manual
portion of the corpus beyond news. This writes into its own shard directory
(``manual_wiki/``, separate from ``manual_news/``) so the two collectors' shard
indices never collide if ever run in the same process.

Title listing (``list=allpages``) is walked sequentially since each page of results
depends on the previous page's continuation cursor, but the per-title extract fetches
within a batch run concurrently across a small thread pool -- the API has no declared
rate limit for anonymous read requests at this volume, and running fetches serially
with a fixed per-request sleep was the main throughput bottleneck during initial runs.
"""

from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import requests

from .record import METHOD_MANUAL, Record, ShardWriter

API_TEMPLATE = "https://{code}.wikipedia.org/w/api.php"
USER_AGENT = (
    "LMA-IndividualProject-Crawler/1.0 "
    "(academic monolingual corpus collection; contact: taufiquesir96@gmail.com)"
)
MIN_ARTICLE_CHARS = 200
#: Wikimedia's anonymous API tier 429s well before this many concurrent readers --
#: found the hard way, see the module docstring history. 4 stays comfortably under it.
FETCH_WORKERS = 4


def _session() -> requests.Session:
    s = requests.Session()
    s.headers["User-Agent"] = USER_AGENT
    return s


def _get_with_backoff(session: requests.Session, api_url: str, params: dict, retries: int = 4) -> requests.Response | None:
    """GET the MediaWiki API, retrying a 429 or transport error with backoff.

    The anonymous API tier rate-limits by request burst, not just steady-state rate --
    a 429 here means "slow down," not "this request is invalid," so it is worth
    retrying rather than dropping the title. A bare ``requests.RequestException``
    (timeout, connection reset) is retried the same way: this call runs inside a
    ThreadPoolExecutor future, and an uncaught exception here used to propagate out of
    the pool and crash the whole ingest the first time any single request timed out --
    found the hard way after both language runs died silently overnight on a
    ReadTimeout with no retry.
    """
    for attempt in range(retries):
        try:
            resp = session.get(api_url, params=params, timeout=20)
        except requests.RequestException:
            time.sleep(2 ** attempt)
            continue
        if resp.status_code != 429:
            return resp
        time.sleep(2 ** attempt)
    return None


def _list_titles(session: requests.Session, api_url: str, gapfrom: str | None, batch: int = 100) -> tuple[list[str] | None, str | None]:
    """Return one batch of article titles via ``list=allpages``, plus a continuation cursor.

    Returns ``(None, gapfrom)`` -- titles is ``None``, not ``[]`` -- when every retry in
    :func:`_get_with_backoff` failed, so the caller can tell "transient failure, retry
    from the same cursor" apart from "the API genuinely returned zero titles" (which
    only happens at the true end of the namespace). Conflating the two was a real bug
    here: a single failed request used to look identical to reaching the end of the
    wiki, silently truncating collection at whatever cursor position the failure
    happened to land on.
    """
    params = {
        "action": "query",
        "list": "allpages",
        "aplimit": batch,
        "apnamespace": 0,
        "apfilterredir": "nonredirects",
        "format": "json",
    }
    if gapfrom:
        params["apfrom"] = gapfrom
    resp = _get_with_backoff(session, api_url, params)
    if resp is None:
        return None, gapfrom
    resp.raise_for_status()
    payload = resp.json()
    titles = [p["title"] for p in payload.get("query", {}).get("allpages", [])]
    cont = payload.get("continue", {}).get("apcontinue")
    return titles, cont


def _fetch_extract(session: requests.Session, api_url: str, title: str) -> str | None:
    """Fetch one article's plain-text extract (no markup, no infobox/reference noise)."""
    params = {
        "action": "query",
        "prop": "extracts",
        "explaintext": 1,
        "titles": title,
        "format": "json",
    }
    resp = _get_with_backoff(session, api_url, params)
    if resp is None or resp.status_code != 200:
        return None
    pages = resp.json().get("query", {}).get("pages", {})
    for page in pages.values():
        text = page.get("extract")
        if text:
            return text
    return None


def run_wikipedia_ingest(lang: str, wiki_code: str, data_root: Path, target_bytes: int) -> int:
    """Pull articles from ``<wiki_code>.wikipedia.org`` until ``target_bytes`` is reached.

    Resumable via a ``wiki_cursor.txt`` file recording the last ``apcontinue`` token
    handed out by the API, so re-running continues alphabetically rather than
    restarting the title walk from the top.

    Returns:
        Total bytes collected this run (new documents only).
    """
    lang_dir = Path(data_root) / lang
    out_dir = lang_dir / "manual_wiki"
    from .record import iter_records

    already = sum(r.n_bytes for r in iter_records(out_dir))
    cursor_path = lang_dir / "wiki_cursor.txt"
    cursor = cursor_path.read_text("utf-8").strip() or None if cursor_path.exists() else None

    api_url = API_TEMPLATE.format(code=wiki_code)
    list_session = _session()
    writer = ShardWriter(out_dir)
    collected_bytes = 0
    collected_docs = 0

    print(f"[{lang}] wikipedia ({wiki_code}.wikipedia.org): {already/1e6:.1f}MB already collected", flush=True)
    try:
        with ThreadPoolExecutor(max_workers=FETCH_WORKERS) as pool:
            fetch_sessions = [_session() for _ in range(FETCH_WORKERS)]
            consecutive_failures = 0
            while already + collected_bytes < target_bytes:
                titles, new_cursor = _list_titles(list_session, api_url, cursor)
                if titles is None:
                    # Every retry in _get_with_backoff failed -- this is not the end of
                    # the namespace, so keep the cursor unchanged and try again rather
                    # than treating it as "exhausted" (see _list_titles' docstring for
                    # why that distinction matters -- it silently truncated a prior run).
                    consecutive_failures += 1
                    if consecutive_failures >= 5:
                        print(f"[{lang}] wikipedia: giving up after 5 consecutive listing failures at cursor {cursor!r}", flush=True)
                        break
                    time.sleep(5 * consecutive_failures)
                    continue
                consecutive_failures = 0
                cursor = new_cursor
                if not titles:
                    print(f"[{lang}] wikipedia: exhausted allpages listing", flush=True)
                    break

                futures = {
                    pool.submit(_fetch_extract, fetch_sessions[i % FETCH_WORKERS], api_url, title): title
                    for i, title in enumerate(titles)
                }
                for future in as_completed(futures):
                    if already + collected_bytes >= target_bytes:
                        break
                    text = future.result()
                    if not text or len(text) < MIN_ARTICLE_CHARS:
                        continue
                    record = Record.build(text=text, domain=f"{wiki_code}.wikipedia.org", method=METHOD_MANUAL, url=futures[future])
                    writer.write(record)
                    writer.flush()
                    collected_bytes += record.n_bytes
                    collected_docs += 1

                cursor_path.write_text(cursor or "", encoding="utf-8")
                print(f"[{lang}] wikipedia: {collected_docs:,} docs, {collected_bytes/1e6:.1f} MB this run", flush=True)
                if not cursor:
                    break
    finally:
        writer.close_shard()
    return collected_bytes
