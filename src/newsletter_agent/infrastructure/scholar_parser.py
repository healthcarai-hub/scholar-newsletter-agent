from __future__ import annotations

import base64
import re
from datetime import datetime, timezone
from html import unescape
from typing import Any, Iterable
from urllib.parse import parse_qs, unquote, urlencode, urlparse, urlunparse

from bs4 import BeautifulSoup, Tag

from newsletter_agent.domain.models import ResearchItem


DOI_PATTERN = re.compile(r"10\.\d{4,9}/[-._;()/:A-Z0-9]+", re.IGNORECASE)
BLOCKED_HOSTS = {"scholar.google.com", "accounts.google.com", "support.google.com"}


class ScholarAlertParseError(ValueError):
    pass


def decode_gmail_body(data: str | None) -> str:
    if not data:
        return ""
    padding = "=" * (-len(data) % 4)
    return base64.urlsafe_b64decode(data + padding).decode("utf-8", errors="replace")


def extract_mime_bodies(payload: dict[str, Any]) -> tuple[str, str]:
    html_parts: list[str] = []
    text_parts: list[str] = []

    def visit(part: dict[str, Any]) -> None:
        mime_type = part.get("mimeType", "")
        body = decode_gmail_body(part.get("body", {}).get("data"))
        if body and mime_type == "text/html":
            html_parts.append(body)
        elif body and mime_type == "text/plain":
            text_parts.append(body)
        for child in part.get("parts", []) or []:
            visit(child)

    visit(payload)
    return "\n".join(html_parts), "\n".join(text_parts)


def unwrap_google_redirect(url: str) -> str:
    value = unescape(url).strip()
    parsed = urlparse(value)
    query = parse_qs(parsed.query)
    if parsed.netloc.endswith("google.com") and parsed.path in {
        "/url",
        "/scholar_url",
        "/searchurl/rr.html",
    }:
        target = query.get("q", query.get("url", [value]))[0]
        return unquote(target)
    return value


def canonicalize_url(url: str) -> str:
    parsed = urlparse(unwrap_google_redirect(url))
    if parsed.scheme not in {"http", "https"}:
        raise ValueError("only HTTP(S) research links are supported")
    host = (parsed.hostname or "").lower()
    if not host:
        raise ValueError("research link is missing a hostname")
    query = parse_qs(parsed.query, keep_blank_values=True)
    kept: list[tuple[str, str]] = []
    for key in sorted(query):
        if key.lower().startswith("utm_") or key.lower() in {"gclid", "fbclid", "source", "sa"}:
            continue
        for value in sorted(query[key]):
            kept.append((key, value))
    return urlunparse(
        (
            parsed.scheme.lower(),
            parsed.netloc.lower(),
            parsed.path.rstrip("/") or "/",
            "",
            urlencode(kept),
            "",
        )
    )


def extract_doi(*values: str | None) -> str | None:
    for value in values:
        if not value:
            continue
        match = DOI_PATTERN.search(value)
        if match:
            return match.group(0).rstrip(".,;) ]").lower()
    return None


def _clean(value: str) -> str:
    return " ".join(unescape(value).split())


def _is_candidate(anchor: Tag) -> bool:
    href = unwrap_google_redirect(str(anchor.get("href", "")))
    parsed = urlparse(href)
    if parsed.scheme not in {"http", "https"} or not parsed.hostname:
        return False
    if parsed.hostname.lower() in BLOCKED_HOSTS:
        return False
    text = _clean(anchor.get_text(" ", strip=True))
    if len(text) < 8:
        return False
    excluded = {"unsubscribe", "view all", "edit alert", "cancel alert", "help"}
    return text.casefold() not in excluded


def _bounded_scholar_text(anchor: Tag) -> tuple[str, str | None] | None:
    """Read only the metadata blocks belonging to one Scholar result heading.

    Current Scholar alert emails place every result directly in a shared outer
    ``div`` as ``h3`` + author ``div`` + ``div.gse_alrt_sni``. Treating that
    shared div as a result container makes the first result absorb every later
    title and snippet, which can corrupt summaries and create false duplicates.
    """
    heading = anchor.find_parent("h3")
    if heading is None:
        return None

    blocks: list[str] = []
    scholar_snippets: list[str] = []
    for sibling in heading.next_siblings:
        if not isinstance(sibling, Tag):
            continue
        # A heading is the boundary of the next result. Scholar also uses a
        # trailing empty h3, which is an equally safe stopping point.
        if sibling.name == "h3" or sibling.select_one("a.gse_alrt_title[href]"):
            break
        if sibling.name == "br":
            break
        value = _clean(sibling.get_text(" ", strip=True))
        if not value:
            continue
        blocks.append(value)
        classes = {str(value) for value in sibling.get("class", [])}
        if "gse_alrt_sni" in classes:
            scholar_snippets.append(value)

    if not blocks:
        return "", None
    authors = blocks[0][:500]
    # Prefer Scholar's explicit snippet block. The generic second block keeps
    # compatibility with simpler/older alert markup without crossing a result
    # heading boundary.
    snippet_parts = scholar_snippets or blocks[1:2]
    return " ".join(snippet_parts)[:3000], authors


def _nearby_text(anchor: Tag) -> tuple[str, str | None]:
    bounded = _bounded_scholar_text(anchor)
    if bounded is not None:
        return bounded

    # Conservative fallback for layouts where one result is wrapped in its own
    # row or list item. Avoid a generic div/td ancestor: those frequently wrap
    # the entire alert and therefore contain unrelated results.
    container = anchor.find_parent(["li", "tr"])
    if not container:
        return "", None
    title = _clean(anchor.get_text(" ", strip=True))
    full = _clean(container.get_text(" ", strip=True))
    remainder = full.replace(title, "", 1).strip(" -–—·")
    return remainder[:3000], None


def parse_scholar_alert(
    *,
    html: str,
    text: str,
    message_id: str,
) -> tuple[ResearchItem, ...]:
    soup = BeautifulSoup(html or "", "html.parser")
    anchors: Iterable[Tag] = soup.select("h3 a[href], a.gse_alrt_title[href]")
    anchors = list(anchors)
    if not anchors:
        anchors = [anchor for anchor in soup.find_all("a", href=True) if _is_candidate(anchor)]

    items: list[ResearchItem] = []
    seen: set[tuple[str, str]] = set()
    for anchor in anchors:
        if not _is_candidate(anchor):
            continue
        title = _clean(anchor.get_text(" ", strip=True))
        try:
            url = canonicalize_url(str(anchor.get("href")))
        except ValueError:
            continue
        identity = (title.casefold(), url)
        if identity in seen:
            continue
        seen.add(identity)
        snippet, authors = _nearby_text(anchor)
        items.append(
            ResearchItem(
                title=title,
                url=url,
                snippet=snippet,
                authors=authors,
                doi=extract_doi(url, snippet),
                source_message_id=message_id,
            )
        )

    if not items and text.strip():
        # A conservative plain-text fallback: title on one line followed by an HTTP URL.
        pattern = re.compile(r"(?m)^(.{8,300})\n(https?://\S+)$")
        for match in pattern.finditer(text):
            title, url = _clean(match.group(1)), match.group(2).strip()
            try:
                canonical = canonicalize_url(url)
            except ValueError:
                continue
            items.append(
                ResearchItem(
                    title=title,
                    url=canonical,
                    snippet="",
                    source_message_id=message_id,
                    doi=extract_doi(canonical),
                )
            )

    if not items:
        raise ScholarAlertParseError(
            f"labeled Gmail message {message_id} contained no recognizable Scholar results"
        )
    return tuple(items)


def gmail_internal_date(value: str | int | None) -> datetime:
    if value is None:
        return datetime.now(timezone.utc)
    return datetime.fromtimestamp(int(value) / 1000, tz=timezone.utc)
