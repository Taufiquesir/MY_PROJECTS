"""Sitemap-driven news crawler for the manually-collected slice of each corpus.

The assignment requires >=20% of final training tokens to come from manual
collection. Reaching that from news sites means walking article-level XML sitemaps
(or date-partitioned archives for outlets that expose one) rather than paging through
category listings, since listing pages mostly repeat short teasers.

Design choices, and why:

- Threaded, not asyncio. asyncio/httpx can give slightly better throughput per core,
  but this machine has ~7GB RAM and no swap headroom to spare; a small
  ``ThreadPoolExecutor`` against the stdlib ``requests`` session keeps memory flat
  and is simple enough to reason about mid-crawl.
- 2.5 requests/second per domain by default (raised to match any longer
  ``Crawl-delay`` declared in robots.txt -- none of the domains configured here
  declare one, verified by hand before picking this default), enforced by
  :class:`RateLimiter`. Multiple domains run concurrently (one thread per domain,
  each with its own rate limiter), so total throughput scales with domain count.
- Resumable: every fetched URL is appended to ``<lang>/manual_news/done.txt`` only
  *after* its document (if any) is flushed to disk, so a crash can cause a URL to be
  refetched but never silently drops an already-collected document.
"""

from __future__ import annotations

import gzip
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import date, timedelta
from pathlib import Path
from urllib.parse import urljoin
from urllib.robotparser import RobotFileParser

import requests
import trafilatura
from lxml import etree

from .record import METHOD_MANUAL, Record, ShardWriter

#: Identifies this crawler to site operators, with a contact address per courtesy norms.
USER_AGENT = (
    "LMA-IndividualProject-Crawler/1.0 "
    "(academic monolingual corpus collection; contact: taufiquesir96@gmail.com)"
)

MIN_ARTICLE_CHARS = 200


@dataclass
class SiteSpec:
    """Crawl configuration for one news domain, loaded from ``sources_<lang>.yaml``.

    Attributes:
        name: Hostname, also stored as ``Record.domain`` for every doc from this site.
        base_url: Site root, used to resolve robots.txt and relative sitemap links.
        sitemaps: Explicit sitemap URLs to start from.
        date_sitemap: ``str.format`` template with ``{yyyy}``/``{mm}``/``{dd}`` slots,
            expanded into one URL per day across ``[date_start, date_end]``, for sites
            that publish a date-partitioned archive instead of one rolling sitemap.
        url_must_match / url_must_not_match: Optional allow/deny regexes over article
            URLs, to skip tag/category pages a sitemap lists alongside real articles.
        sitemap_must_match: Optional regex a *nested* sitemap URL must satisfy before
            it is followed, for WordPress-style sites that also expose author/tag
            sitemaps we don't want to spend requests walking.
        crawl_delay: Seconds between requests to this domain (floor; robots.txt can
            raise it).
        max_urls: Cap on URLs discovered from this domain.
    """

    name: str
    base_url: str
    sitemaps: list[str] = field(default_factory=list)
    date_sitemap: str | None = None
    date_start: str | None = None
    date_end: str | None = None
    url_must_match: str | None = None
    url_must_not_match: str | None = None
    sitemap_must_match: str | None = None
    crawl_delay: float = 0.4
    max_urls: int = 150_000

    def dated_sitemap_urls(self) -> list[str]:
        """Expand ``date_sitemap`` into one URL per day, most recent first."""
        if not (self.date_sitemap and self.date_start and self.date_end):
            return []
        start, end = date.fromisoformat(self.date_start), date.fromisoformat(self.date_end)
        out, cur = [], end
        while cur >= start:
            out.append(self.date_sitemap.format(yyyy=f"{cur.year:04d}", mm=f"{cur.month:02d}", dd=f"{cur.day:02d}"))
            cur -= timedelta(days=1)
        return out


class RateLimiter:
    """Caps request *starts* to one per ``delay`` seconds, thread-safe."""

    def __init__(self, delay: float) -> None:
        self.delay = delay
        self._next_ok = 0.0
        self._lock = threading.Lock()

    def wait(self) -> None:
        with self._lock:
            now = time.monotonic()
            sleep_for = max(0.0, self._next_ok - now)
            self._next_ok = max(now, self._next_ok) + self.delay
        if sleep_for:
            time.sleep(sleep_for)


class CrawlBudget:
    """Shared, thread-safe byte budget and resume log for one language's news crawl."""

    def __init__(self, lang_dir: Path, target_bytes: int, seed_bytes: int = 0) -> None:
        self.lang_dir = Path(lang_dir)
        self.lang_dir.mkdir(parents=True, exist_ok=True)
        self.done_path = self.lang_dir / "done.txt"
        self.done: set[str] = set()
        if self.done_path.exists():
            self.done = {l.strip() for l in self.done_path.read_text("utf-8").splitlines() if l.strip()}
        self._done_fh = self.done_path.open("a", encoding="utf-8")
        self.target_bytes = target_bytes
        self.bytes_collected = seed_bytes
        self.docs_collected = 0
        self._lock = threading.Lock()
        self._last_report = time.monotonic()

    def exhausted(self) -> bool:
        return self.bytes_collected >= self.target_bytes

    def record(self, writer: ShardWriter, url: str, record: Record | None) -> None:
        with self._lock:
            if record is not None:
                writer.write(record)
                writer.flush()
                self.bytes_collected += record.n_bytes
                self.docs_collected += 1
            self._done_fh.write(url + "\n")
            self._done_fh.flush()
            self.done.add(url)
            now = time.monotonic()
            if now - self._last_report > 20.0:
                pct = 100 * self.bytes_collected / self.target_bytes if self.target_bytes else 0
                print(
                    f"[progress] {self.docs_collected:,} docs, "
                    f"{self.bytes_collected / 1e6:.1f} MB ({pct:.1f}% of target)",
                    flush=True,
                )
                self._last_report = now

    def close(self) -> None:
        self._done_fh.close()


def _local_tag(tag: object) -> str:
    if not isinstance(tag, str):
        return ""
    return tag.rsplit("}", 1)[-1]


def _parse_sitemap_xml(content: bytes) -> tuple[list[str], list[str]]:
    """Split a sitemap document into (nested sitemap URLs, article/page URLs).

    Namespace-agnostic on purpose: real sites are inconsistent about which sitemap
    namespace URI they declare, so classification is done by each ``<loc>``'s parent
    tag name rather than by binding to one namespace.
    """
    if content[:2] == b"\x1f\x8b":
        try:
            content = gzip.decompress(content)
        except OSError:
            return [], []
    try:
        root = etree.fromstring(content, parser=etree.XMLParser(recover=True, huge_tree=True))
    except etree.XMLSyntaxError:
        return [], []
    if root is None:
        return [], []
    sitemaps, pages = [], []
    for el in root.iter():
        if _local_tag(el.tag) != "loc" or not el.text:
            continue
        parent = el.getparent()
        (sitemaps if _local_tag(parent.tag) == "sitemap" else pages).append(el.text.strip())
    return sitemaps, pages


def _fetch(session: requests.Session, url: str, timeout: float = 20.0) -> requests.Response | None:
    try:
        resp = session.get(url, timeout=timeout)
    except requests.RequestException:
        return None
    return resp if resp.status_code == 200 else None


def _load_robots(session: requests.Session, spec: SiteSpec) -> RobotFileParser | None:
    resp = _fetch(session, urljoin(spec.base_url, "/robots.txt"))
    if resp is None:
        return None
    parser = RobotFileParser()
    parser.parse(resp.text.splitlines())
    declared = parser.crawl_delay(USER_AGENT)
    if declared is not None:
        spec.crawl_delay = max(spec.crawl_delay, float(declared))
    return parser


def _discover_urls(session: requests.Session, spec: SiteSpec, robots: RobotFileParser | None, limiter: RateLimiter) -> list[str]:
    """Walk sitemap indexes breadth-first and return up to ``spec.max_urls`` article URLs."""
    queue = list(spec.sitemaps) + spec.dated_sitemap_urls()
    if not queue and robots is not None:
        queue = list(robots.site_maps() or [])
    if not queue:
        queue = [urljoin(spec.base_url, "/sitemap.xml")]

    allow = re.compile(spec.url_must_match) if spec.url_must_match else None
    deny = re.compile(spec.url_must_not_match) if spec.url_must_not_match else None
    sm_allow = re.compile(spec.sitemap_must_match) if spec.sitemap_must_match else None
    roots = set(queue)
    seen_sitemaps: set[str] = set()
    seen_urls: list[str] = []
    seen_set: set[str] = set()
    walked = 0

    while queue and len(seen_urls) < spec.max_urls:
        sm_url = queue.pop(0)
        if sm_url in seen_sitemaps:
            continue
        seen_sitemaps.add(sm_url)
        if sm_allow is not None and sm_url not in roots and not sm_allow.search(sm_url):
            continue
        limiter.wait()
        resp = _fetch(session, sm_url)
        walked += 1
        if resp is None:
            continue
        nested, pages = _parse_sitemap_xml(resp.content)
        queue.extend(nested)
        for u in pages:
            if len(seen_urls) >= spec.max_urls or u in seen_set:
                continue
            if allow is not None and not allow.search(u):
                continue
            if deny is not None and deny.search(u):
                continue
            if robots is not None and not robots.can_fetch(USER_AGENT, u):
                continue
            seen_set.add(u)
            seen_urls.append(u)
        if walked % 50 == 0:
            print(f"[{spec.name}] walked {walked} sitemaps, {len(seen_urls):,} URLs so far", flush=True)
    return seen_urls


def _fetch_article(session: requests.Session, spec: SiteSpec, url: str, budget: CrawlBudget, writer: ShardWriter) -> None:
    resp = _fetch(session, url)
    if resp is None:
        budget.record(writer, url, None)
        return
    text = trafilatura.extract(resp.text, include_comments=False, include_tables=False)
    record = None
    if text and len(text) >= MIN_ARTICLE_CHARS:
        record = Record.build(text=text, domain=spec.name, method=METHOD_MANUAL, url=url)
    budget.record(writer, url, record)


def crawl_site(spec: SiteSpec, budget: CrawlBudget, writer: ShardWriter, workers: int = 10) -> None:
    """Discover and fetch one domain's articles until the shared budget is exhausted."""
    session = requests.Session()
    session.headers["User-Agent"] = USER_AGENT
    robots = _load_robots(session, spec)
    limiter = RateLimiter(spec.crawl_delay)
    print(f"[{spec.name}] crawl_delay={spec.crawl_delay}s; discovering URLs", flush=True)
    urls = _discover_urls(session, spec, robots, limiter)
    todo = [u for u in urls if u not in budget.done]
    print(f"[{spec.name}] {len(urls):,} URLs discovered, {len(todo):,} not yet fetched", flush=True)

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = []
        for url in todo:
            if budget.exhausted():
                break
            limiter.wait()
            futures.append(pool.submit(_fetch_article, session, spec, url, budget, writer))
        for f in as_completed(futures):
            f.result()


def run_news_crawl(lang: str, specs: list[SiteSpec], data_root: Path, target_bytes: int) -> CrawlBudget:
    """Crawl every configured domain for one language into ``<data_root>/<lang>/manual_news``.

    Domains are crawled *concurrently*, one thread per domain, each domain keeping its
    own rate limiter (so politeness is still one-request-per-second-per-domain) and its
    own :class:`ShardWriter` pointed at a per-domain subdirectory -- this avoids two
    writers racing on the same shard-index counter, which would otherwise let one
    domain's shard-seal silently clobber another's in-progress shard. Running domains
    in parallel rather than one-after-another is what lets total throughput scale with
    the number of configured domains instead of being capped at a single domain's
    1 req/s.
    """
    lang_dir = Path(data_root) / lang
    out_dir = lang_dir / "manual_news"
    from .record import iter_records

    seed_bytes = sum(r.n_bytes for r in iter_records(out_dir))
    budget = CrawlBudget(lang_dir, target_bytes, seed_bytes=seed_bytes)
    writers = {spec.name: ShardWriter(out_dir / spec.name) for spec in specs}
    print(f"[{lang}] resuming with {budget.bytes_collected / 1e6:.1f} MB already collected, {len(budget.done):,} URLs done", flush=True)
    try:
        with ThreadPoolExecutor(max_workers=len(specs)) as domain_pool:
            futures = [domain_pool.submit(crawl_site, spec, budget, writers[spec.name]) for spec in specs]
            for f in as_completed(futures):
                f.result()
    finally:
        for w in writers.values():
            w.close_shard()
        budget.close()
    return budget
