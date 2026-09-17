from __future__ import annotations

import re
from collections import defaultdict
from dataclasses import replace
from difflib import SequenceMatcher
from urllib.parse import urlparse

from newsletter_agent.config import ContentPolicyConfig
from newsletter_agent.domain.models import ClassifiedItem, EnrichedItem
from newsletter_agent.infrastructure.enrichment import normalize_identity_text


SOURCE_TYPES = {
    "research_paper",
    "academic_review",
    "conference_material",
    "research_news",
    "commercial_blog",
    "marketing_page",
    "seo_content",
    "unknown",
}

_ACADEMIC_HOST_PARTS = (
    "pubmed.ncbi.nlm.nih.gov",
    "ncbi.nlm.nih.gov",
    "sciencedirect.com",
    "springer.com",
    "wiley.com",
    "tandfonline.com",
    "mdpi.com",
    "nature.com",
    "science.org",
)
_SEO_TITLE = re.compile(
    r"\b(best|top\s+\d+|ranked|complete guide|what actually works|how fast do|"
    r"supplement guide|brain booster supplements?)\b",
    re.IGNORECASE,
)
_COMMERCIAL = re.compile(
    r"\b(buy now|add to cart|shop now|discount|coupon|product variants?|supplement stack|"
    r"best supplement|pre workout|nootropic drinks?)\b",
    re.IGNORECASE,
)
_BLOG_PATH = re.compile(r"/(blogs?|articles?|guides?)/", re.IGNORECASE)


def infer_source_type(item: EnrichedItem, model_suggestion: str = "unknown") -> str:
    """Combine durable source evidence with a constrained model suggestion."""
    parsed = urlparse(item.canonical_url)
    host = (parsed.hostname or "").casefold()
    path = parsed.path.casefold()
    title = item.item.title
    # URL slugs often preserve commercial intent even when Scholar supplies a
    # generic or truncated title (for example ``pre-workout-review``).
    evidence = " ".join((title, item.item.snippet, item.abstract or "", path.replace("-", " ")))

    # Durable scholarly identifiers outrank stylistic signals and access failures.
    if item.doi or any(part in host for part in _ACADEMIC_HOST_PARTS):
        if "conference" in evidence.casefold() or "proceedings" in evidence.casefold():
            return "conference_material"
        if "review" in title.casefold() or "meta-analysis" in title.casefold():
            return "academic_review"
        return "research_paper"
    if item.venue and item.authors:
        return "research_paper"
    if "/product" in path or _COMMERCIAL.search(evidence):
        return "marketing_page"
    if _BLOG_PATH.search(path) and _SEO_TITLE.search(title):
        return "seo_content"
    if _BLOG_PATH.search(path):
        return "commercial_blog" if _COMMERCIAL.search(evidence) else "research_news"

    suggestion = model_suggestion.strip().casefold().replace("-", "_").replace(" ", "_")
    return suggestion if suggestion in SOURCE_TYPES else "unknown"


def with_source_quality(
    item: ClassifiedItem,
    model_suggestion: str,
    low_priority_source_types: tuple[str, ...] = (
        "commercial_blog",
        "marketing_page",
        "seo_content",
    ),
) -> ClassifiedItem:
    source_type = infer_source_type(item.enriched, model_suggestion)
    priority = "low" if source_type in set(low_priority_source_types) else "normal"
    return replace(item, source_type=source_type, priority=priority)


def _tokens(value: str) -> set[str]:
    return set(normalize_identity_text(value).split())


def _title_similarity(left: ClassifiedItem, right: ClassifiedItem) -> float:
    a, b = _tokens(left.enriched.item.title), _tokens(right.enriched.item.title)
    if not a or not b:
        return 0.0
    containment = len(a & b) / min(len(a), len(b))
    jaccard = len(a & b) / len(a | b)
    return max(containment, jaccard)


def _content_similarity(left: ClassifiedItem, right: ClassifiedItem) -> float:
    a = normalize_identity_text(left.enriched.abstract or left.enriched.item.snippet)
    b = normalize_identity_text(right.enriched.abstract or right.enriched.item.snippet)
    if min(len(a.split()), len(b.split())) < 12:
        return 0.0
    return SequenceMatcher(None, a, b, autojunk=False).ratio()


def _same_low_priority_cluster(
    left: ClassifiedItem, right: ClassifiedItem, policy: ContentPolicyConfig
) -> bool:
    if left.priority != "low" or right.priority != "low":
        return False
    left_host = (urlparse(left.enriched.canonical_url).hostname or "").casefold()
    right_host = (urlparse(right.enriched.canonical_url).hostname or "").casefold()
    if policy.cluster_within_same_domain and left_host != right_host:
        return False
    return (
        _title_similarity(left, right) >= policy.title_similarity_threshold
        or _content_similarity(left, right) >= policy.content_similarity_threshold
    )


def _representative_score(item: ClassifiedItem, index: int) -> tuple[int, ...]:
    source_words = len(item.enriched.source_text.split())
    return (
        int(item.enriched.link_status == "available"),
        int(bool(item.enriched.abstract)),
        int(bool(item.enriched.authors)),
        source_words,
        -index,
    )


def apply_content_policy(
    items: list[ClassifiedItem], policy: ContentPolicyConfig
) -> list[ClassifiedItem]:
    """Collapse redundant low-priority clusters and render low-priority items last."""
    if not policy.enabled or len(items) < 2:
        return items.copy()

    def host_for(item: ClassifiedItem) -> str:
        return (urlparse(item.enriched.canonical_url).hostname or "").casefold()

    academic_types = {"research_paper", "academic_review", "conference_material"}
    low_priority_domains = {host_for(item) for item in items if item.priority == "low"}
    # Commercial platforms can mix obvious SEO pages with pages the model calls
    # research news. Once a domain exhibits strong low-priority evidence, treat
    # all non-academic material from that platform consistently. Durable academic
    # source types remain exempt.
    items = [
        replace(item, priority="low")
        if host_for(item) in low_priority_domains and item.source_type not in academic_types
        else item
        for item in items
    ]

    parents = list(range(len(items)))

    def find(index: int) -> int:
        while parents[index] != index:
            parents[index] = parents[parents[index]]
            index = parents[index]
        return index

    def union(left: int, right: int) -> None:
        a, b = find(left), find(right)
        if a != b:
            parents[b] = a

    for left_index, left in enumerate(items):
        for right_index in range(left_index + 1, len(items)):
            if _same_low_priority_cluster(left, items[right_index], policy):
                union(left_index, right_index)

    clusters: dict[int, list[int]] = defaultdict(list)
    for index in range(len(items)):
        clusters[find(index)].append(index)

    kept_indices: set[int] = set()
    for indices in clusters.values():
        ranked = sorted(
            indices,
            key=lambda index: _representative_score(items[index], index),
            reverse=True,
        )
        limit = policy.max_items_per_cluster if items[indices[0]].priority == "low" else len(ranked)
        kept_indices.update(ranked[:limit])

    retained = [item for index, item in enumerate(items) if index in kept_indices]

    # After thematic clustering, apply a newsletter-level diversity cap to the
    # remaining low-priority pool. Academic/research items are deliberately
    # exempt so a publisher can contribute multiple distinct papers.
    low_by_domain: dict[str, list[tuple[int, ClassifiedItem]]] = defaultdict(list)
    for index, item in enumerate(retained):
        if item.priority != "low":
            continue
        host = host_for(item)
        low_by_domain[host].append((index, item))
    remove_ids: set[int] = set()
    for domain_items in low_by_domain.values():
        ranked = sorted(
            domain_items,
            key=lambda pair: _representative_score(pair[1], pair[0]),
            reverse=True,
        )
        remove_ids.update(
            id(item)
            for _, item in ranked[policy.max_low_priority_items_per_domain :]
        )
    retained = [item for item in retained if id(item) not in remove_ids]
    return sorted(retained, key=lambda item: item.priority == "low")
