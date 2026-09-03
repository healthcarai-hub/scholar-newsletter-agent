from __future__ import annotations

import asyncio
import hashlib
import ipaddress
import json
import re
import socket
import unicodedata
from collections import defaultdict
from difflib import SequenceMatcher
from html import unescape
from typing import Any
from urllib.parse import urljoin, urlparse

from bs4 import BeautifulSoup

from newsletter_agent.domain.models import EnrichedItem, ResearchItem
from newsletter_agent.infrastructure.scholar_parser import canonicalize_url, extract_doi


class UnsafeURL(ValueError):
    pass


_IDENTITY_WORDS = re.compile(r"[^\W_]+", re.UNICODE)
_SOFT_404_MARKERS = (
    "404 not found",
    "404 page not found",
    "article not found",
    "content not found",
    "page does not exist",
    "page not found",
    "requested page could not be found",
    "resource not found",
)


def normalize_identity_text(value: str | None) -> str:
    """Normalize human-readable source text for conservative identity matching."""
    if not value:
        return ""
    normalized = unicodedata.normalize("NFKC", unescape(value)).casefold()
    return " ".join(_IDENTITY_WORDS.findall(normalized))


def looks_like_soft_404(html: str) -> bool:
    """Detect common HTTP-200 error pages without scanning research body text."""
    soup = BeautifulSoup(html, "html.parser")
    candidates = []
    if soup.title:
        candidates.append(soup.title.get_text(" ", strip=True))
    candidates.extend(
        heading.get_text(" ", strip=True) for heading in soup.find_all(["h1", "h2"], limit=3)
    )
    normalized_candidates = [normalize_identity_text(value) for value in candidates]
    normalized_markers = tuple(normalize_identity_text(marker) for marker in _SOFT_404_MARKERS)
    return any(
        marker in candidate
        for candidate in normalized_candidates
        for marker in normalized_markers
    )


def filter_dead_links(items: list[EnrichedItem]) -> list[EnrichedItem]:
    """Exclude only links conclusively identified as missing."""
    return [item for item in items if item.link_status != "dead"]


def _identity_title(item: EnrichedItem) -> str:
    enriched_title = item.metadata.get("title")
    return normalize_identity_text(
        str(enriched_title) if enriched_title else item.item.title
    )


def _content_candidates(item: EnrichedItem) -> tuple[str, ...]:
    candidates = []
    for value in (item.abstract, item.item.snippet):
        normalized = normalize_identity_text(value)
        if normalized and normalized not in candidates:
            candidates.append(normalized)
    return tuple(candidates)


def _word_shingles(value: str, size: int = 3) -> set[tuple[str, ...]]:
    words = value.split()
    if len(words) < size:
        return set()
    return {tuple(words[index : index + size]) for index in range(len(words) - size + 1)}


def _content_matches(left: EnrichedItem, right: EnrichedItem) -> bool:
    for left_text in _content_candidates(left):
        left_words = left_text.split()
        for right_text in _content_candidates(right):
            right_words = right_text.split()
            minimum_words = min(len(left_words), len(right_words))
            if left_text == right_text and minimum_words >= 10:
                return True
            if minimum_words < 18:
                continue

            left_tokens, right_tokens = set(left_words), set(right_words)
            shared_tokens = left_tokens & right_tokens
            containment = len(shared_tokens) / min(len(left_tokens), len(right_tokens))
            if len(shared_tokens) >= 16 and containment >= 0.92:
                return True

            left_shingles = _word_shingles(left_text)
            right_shingles = _word_shingles(right_text)
            union = left_shingles | right_shingles
            shingle_similarity = len(left_shingles & right_shingles) / len(union) if union else 0
            if shingle_similarity >= 0.78:
                return True

            if SequenceMatcher(None, left_text, right_text, autojunk=False).ratio() >= 0.92:
                return True
    return False


def _same_research_result(left: EnrichedItem, right: EnrichedItem) -> bool:
    if left.doi and right.doi and left.doi.casefold() == right.doi.casefold():
        return True
    if left.canonical_url and left.canonical_url == right.canonical_url:
        return True

    left_title, right_title = _identity_title(left), _identity_title(right)
    if left_title == right_title and (len(left_title) >= 30 or len(left_title.split()) >= 4):
        return True
    return _content_matches(left, right)


def _representative_score(item: EnrichedItem, index: int) -> tuple[int, ...]:
    content_size = max((len(value.split()) for value in _content_candidates(item)), default=0)
    metadata_size = sum(bool(value) for value in item.metadata.values())
    return (
        int(bool(item.doi)),
        int(bool(item.abstract)),
        int(item.enrichment_status == "enriched"),
        content_size,
        metadata_size,
        -index,
    )


def normalized_source_text(
    *,
    title: str,
    snippet: str = "",
    abstract: str = "",
    authors: str = "",
    venue: str = "",
    published_at: str = "",
    doi: str = "",
) -> str:
    fields = [
        ("Title", title),
        ("Authors", authors),
        ("Venue", venue),
        ("Published", published_at),
        ("DOI", doi),
        ("Abstract", abstract),
        ("Scholar snippet", snippet),
    ]
    return "\n".join(f"{label}: {' '.join(value.split())}" for label, value in fields if value)


def research_fingerprint(item: ResearchItem, canonical_url: str, doi: str | None = None) -> str:
    if doi:
        identity = f"doi:{doi.casefold()}"
    elif canonical_url:
        identity = f"url:{canonical_url}"
    else:
        identity = f"title:{' '.join(item.title.casefold().split())}"
    return hashlib.sha256(identity.encode("utf-8")).hexdigest()


async def validate_public_url(url: str) -> None:
    parsed = urlparse(url)
    if parsed.scheme not in {"http", "https"} or not parsed.hostname:
        raise UnsafeURL("only public HTTP(S) URLs can be enriched")
    host = parsed.hostname.casefold()
    if host in {"localhost", "localhost.localdomain"} or host.endswith(".local"):
        raise UnsafeURL("local network URLs are not allowed")
    loop = asyncio.get_running_loop()
    try:
        records = await loop.getaddrinfo(host, parsed.port or (443 if parsed.scheme == "https" else 80), type=socket.SOCK_STREAM)
    except OSError as exc:
        raise UnsafeURL(f"hostname could not be resolved: {host}") from exc
    for record in records:
        address = ipaddress.ip_address(record[4][0])
        if not address.is_global:
            raise UnsafeURL(f"non-public destination is not allowed: {address}")


def _meta_content(soup: BeautifulSoup, *names: str) -> str | None:
    for name in names:
        tag = soup.find("meta", attrs={"name": name}) or soup.find("meta", attrs={"property": name})
        if tag and tag.get("content"):
            value = " ".join(str(tag["content"]).split())
            if value:
                return value
    return None


def extract_page_metadata(html: str) -> dict[str, Any]:
    soup = BeautifulSoup(html, "html.parser")
    data: dict[str, Any] = {
        "title": _meta_content(soup, "citation_title", "dc.title", "og:title"),
        "abstract": _meta_content(
            soup,
            "citation_abstract",
            "dc.description",
            "description",
            "og:description",
        ),
        "venue": _meta_content(soup, "citation_journal_title", "prism.publicationName"),
        "published_at": _meta_content(
            soup, "citation_publication_date", "citation_date", "article:published_time"
        ),
        "doi": _meta_content(soup, "citation_doi", "dc.identifier"),
    }
    authors = [
        " ".join(str(tag.get("content", "")).split())
        for tag in soup.find_all("meta", attrs={"name": "citation_author"})
        if tag.get("content")
    ]
    if authors:
        data["authors"] = ", ".join(authors)

    for script in soup.find_all("script", attrs={"type": "application/ld+json"}):
        try:
            payload = json.loads(script.string or "")
        except (TypeError, json.JSONDecodeError):
            continue
        candidates = payload if isinstance(payload, list) else [payload]
        for candidate in candidates:
            if not isinstance(candidate, dict):
                continue
            data["title"] = data.get("title") or candidate.get("headline") or candidate.get("name")
            data["abstract"] = data.get("abstract") or candidate.get("abstract") or candidate.get("description")
            data["published_at"] = data.get("published_at") or candidate.get("datePublished")
            identifier = candidate.get("identifier")
            if isinstance(identifier, str):
                data["doi"] = data.get("doi") or identifier
    return {key: value for key, value in data.items() if value}


class HttpMetadataEnricher:
    def __init__(
        self,
        *,
        timeout_seconds: float = 15,
        max_page_bytes: int = 2_000_000,
        max_concurrency: int = 6,
        per_domain_concurrency: int = 2,
        max_redirects: int = 4,
    ) -> None:
        self.timeout_seconds = timeout_seconds
        self.max_page_bytes = max_page_bytes
        self.global_semaphore = asyncio.Semaphore(max_concurrency)
        self.per_domain_concurrency = per_domain_concurrency
        self.domain_semaphores: defaultdict[str, asyncio.Semaphore] = defaultdict(
            lambda: asyncio.Semaphore(self.per_domain_concurrency)
        )
        self.max_redirects = max_redirects

    async def _fetch_html(self, url: str) -> tuple[str, str]:
        try:
            import httpx
        except ImportError as exc:
            raise RuntimeError("httpx is required for landing-page enrichment") from exc

        current = url
        headers = {
            "User-Agent": "ScholarNewsletterAgent/0.1 (+research metadata fetcher)",
            "Accept": "text/html,application/xhtml+xml;q=0.9,*/*;q=0.1",
        }
        async with httpx.AsyncClient(timeout=self.timeout_seconds, follow_redirects=False) as client:
            for _ in range(self.max_redirects + 1):
                await validate_public_url(current)
                host = (urlparse(current).hostname or "").casefold()
                async with self.global_semaphore, self.domain_semaphores[host]:
                    async with client.stream("GET", current, headers=headers) as response:
                        if response.status_code in {301, 302, 303, 307, 308}:
                            location = response.headers.get("location")
                            if not location:
                                raise RuntimeError("redirect response did not include a destination")
                            current = urljoin(current, location)
                            continue
                        response.raise_for_status()
                        content_type = response.headers.get("content-type", "").casefold()
                        if "html" not in content_type and "xhtml" not in content_type:
                            raise RuntimeError(f"unsupported landing-page content type: {content_type or 'unknown'}")
                        chunks: list[bytes] = []
                        total = 0
                        async for chunk in response.aiter_bytes():
                            total += len(chunk)
                            if total > self.max_page_bytes:
                                raise RuntimeError("landing page exceeded configured size limit")
                            chunks.append(chunk)
                        encoding = response.encoding or "utf-8"
                        return b"".join(chunks).decode(encoding, errors="replace"), current
            raise RuntimeError("landing page exceeded redirect limit")

    async def enrich(self, item: ResearchItem) -> EnrichedItem:
        canonical = canonicalize_url(item.url)
        metadata: dict[str, Any] = {}
        status = "alert_only"
        link_status = "unverified"
        http_status = None
        try:
            html, final_url = await self._fetch_html(canonical)
            canonical = canonicalize_url(final_url)
            http_status = 200
            if looks_like_soft_404(html):
                link_status = "dead"
                metadata = {"enrichment_error": "Soft404", "http_status": 200}
            else:
                metadata = extract_page_metadata(html)
                status = "enriched"
                link_status = "available"
        except Exception as exc:  # network and publisher failures deliberately fall back to the alert
            response = getattr(exc, "response", None)
            response_status = getattr(response, "status_code", None)
            http_status = response_status if isinstance(response_status, int) else None
            link_status = "dead" if http_status in {404, 410} else "unverified"
            metadata = {
                "enrichment_error": type(exc).__name__,
                "link_status": link_status,
            }
            if http_status is not None:
                metadata["http_status"] = http_status

        metadata.setdefault("link_status", link_status)
        if http_status is not None:
            metadata.setdefault("http_status", http_status)

        doi = extract_doi(str(metadata.get("doi", "")), item.doi, canonical, item.snippet)
        authors = str(metadata.get("authors") or item.authors or "") or None
        venue = str(metadata.get("venue") or item.venue or "") or None
        published_at = str(metadata.get("published_at") or item.published_at or "") or None
        abstract = str(metadata.get("abstract") or "") or None
        source_text = normalized_source_text(
            title=str(metadata.get("title") or item.title),
            snippet=item.snippet,
            abstract=abstract or "",
            authors=authors or "",
            venue=venue or "",
            published_at=published_at or "",
            doi=doi or "",
        )
        return EnrichedItem(
            item=item,
            canonical_url=canonical,
            fingerprint=research_fingerprint(item, canonical, doi),
            source_text=source_text,
            abstract=abstract,
            authors=authors,
            venue=venue,
            published_at=published_at,
            doi=doi,
            enrichment_status=status,
            link_status=link_status,
            http_status=http_status,
            metadata=metadata,
        )


def deduplicate_enriched(items: list[EnrichedItem]) -> list[EnrichedItem]:
    """Cluster duplicate results using durable IDs plus conservative content similarity."""
    if len(items) < 2:
        return items.copy()

    parents = list(range(len(items)))

    def find(index: int) -> int:
        while parents[index] != index:
            parents[index] = parents[parents[index]]
            index = parents[index]
        return index

    def union(left: int, right: int) -> None:
        left_root, right_root = find(left), find(right)
        if left_root != right_root:
            parents[right_root] = left_root

    for left_index, left in enumerate(items):
        for right_index in range(left_index + 1, len(items)):
            if left.fingerprint == items[right_index].fingerprint or _same_research_result(
                left, items[right_index]
            ):
                union(left_index, right_index)

    clusters: dict[int, list[int]] = defaultdict(list)
    for index in range(len(items)):
        clusters[find(index)].append(index)

    representatives = []
    for indices in sorted(clusters.values(), key=min):
        best_index = max(indices, key=lambda index: _representative_score(items[index], index))
        representatives.append(items[best_index])
    return representatives
