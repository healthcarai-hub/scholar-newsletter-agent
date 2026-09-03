from __future__ import annotations

import asyncio

import httpx

from newsletter_agent.domain.models import ResearchItem
from newsletter_agent.infrastructure.enrichment import (
    HttpMetadataEnricher,
    deduplicate_enriched,
    extract_page_metadata,
    filter_dead_links,
    looks_like_soft_404,
    normalize_identity_text,
    normalized_source_text,
    research_fingerprint,
)
from newsletter_agent.domain.models import EnrichedItem


def test_extracts_citation_metadata():
    html = """
    <html><head>
      <meta name="citation_title" content="A Trial">
      <meta name="citation_author" content="A. Author">
      <meta name="citation_author" content="B. Author">
      <meta name="citation_abstract" content="  A useful abstract. ">
      <meta name="citation_doi" content="10.1000/example">
    </head></html>
    """
    result = extract_page_metadata(html)
    assert result["title"] == "A Trial"
    assert result["authors"] == "A. Author, B. Author"
    assert result["abstract"] == "A useful abstract."


def test_detects_soft_404_from_page_heading():
    assert looks_like_soft_404(
        "<html><head><title>Publisher</title></head><body><h1>Page not found</h1></body></html>"
    )
    assert not looks_like_soft_404(
        "<html><head><title>A Migraine Trial</title></head><body><h1>Study results</h1></body></html>"
    )


def test_enrichment_marks_http_404_as_dead_and_filters_it():
    item = ResearchItem(
        "Unavailable paper", "https://example.org/missing", "Scholar snippet", "message-1"
    )
    enricher = HttpMetadataEnricher()

    async def missing(_url):
        request = httpx.Request("GET", item.url)
        response = httpx.Response(404, request=request)
        raise httpx.HTTPStatusError("not found", request=request, response=response)

    enricher._fetch_html = missing
    enriched = asyncio.run(enricher.enrich(item))

    assert enriched.link_status == "dead"
    assert enriched.http_status == 404
    assert enriched.metadata["http_status"] == 404
    assert filter_dead_links([enriched]) == []


def test_enrichment_keeps_blocked_link_as_unverified():
    item = ResearchItem(
        "Blocked paper", "https://example.org/blocked", "Scholar snippet", "message-1"
    )
    enricher = HttpMetadataEnricher()

    async def blocked(_url):
        request = httpx.Request("GET", item.url)
        response = httpx.Response(403, request=request)
        raise httpx.HTTPStatusError("forbidden", request=request, response=response)

    enricher._fetch_html = blocked
    enriched = asyncio.run(enricher.enrich(item))

    assert enriched.link_status == "unverified"
    assert enriched.http_status == 403
    assert filter_dead_links([enriched]) == [enriched]


def test_enrichment_marks_soft_404_as_dead():
    item = ResearchItem(
        "Missing article", "https://example.org/soft-missing", "Scholar snippet", "message-1"
    )
    enricher = HttpMetadataEnricher()

    async def soft_missing(url):
        return "<html><head><title>404 Page Not Found</title></head></html>", url

    enricher._fetch_html = soft_missing
    enriched = asyncio.run(enricher.enrich(item))

    assert enriched.link_status == "dead"
    assert enriched.http_status == 200
    assert enriched.metadata["enrichment_error"] == "Soft404"


def test_source_text_is_grounded_and_fingerprint_prefers_doi():
    item = ResearchItem(
        title="A Trial",
        url="https://example.org/a",
        snippet="Alert snippet",
        source_message_id="1",
    )
    text = normalized_source_text(title=item.title, snippet=item.snippet, doi="10.1000/example")
    assert "Title: A Trial" in text
    assert "Scholar snippet: Alert snippet" in text
    assert research_fingerprint(item, item.url, "10.1000/example") == research_fingerprint(
        item, "https://other.example/b", "10.1000/example"
    )


def test_deduplicates_enriched_results():
    item = ResearchItem("A Trial", "https://example.org/a", "", "1")
    enriched = EnrichedItem(item, item.url, "same", "Title: A Trial")
    assert deduplicate_enriched([enriched, enriched]) == [enriched]


def _enriched(
    title: str,
    url: str,
    *,
    snippet: str = "",
    abstract: str | None = None,
    doi: str | None = None,
    status: str = "alert_only",
) -> EnrichedItem:
    original = ResearchItem(title, url, snippet, title, doi=doi)
    return EnrichedItem(
        item=original,
        canonical_url=url,
        fingerprint=research_fingerprint(original, url, doi),
        source_text=normalized_source_text(
            title=title,
            snippet=snippet,
            abstract=abstract or "",
            doi=doi or "",
        ),
        abstract=abstract,
        doi=doi,
        enrichment_status=status,
    )


def test_normalizes_identity_text_across_case_punctuation_and_spacing():
    assert normalize_identity_text("  Migraine—RESEARCH:  A Study! ") == (
        "migraine research a study"
    )


def test_deduplicates_different_titles_and_urls_from_identical_content():
    content = (
        "This randomized controlled study evaluates a preventive migraine treatment "
        "and reports reductions in monthly headache days with comparable safety outcomes."
    )
    first = _enriched("Preventive treatment trial", "https://publisher.example/article", snippet=content)
    second = _enriched("Migraine prevention outcomes", "https://repository.example/record", snippet=content)

    assert deduplicate_enriched([first, second]) == [first]


def test_deduplicates_when_only_one_copy_exposes_doi_using_shared_abstract():
    abstract = (
        "Researchers assessed cortical activity in adults with migraine using repeated "
        "measurements before and after treatment, finding consistent changes across the "
        "primary neurological outcomes included in the analysis."
    )
    without_doi = _enriched(
        "Cortical activity after treatment",
        "https://repository.example/preprint",
        abstract=abstract,
    )
    with_doi = _enriched(
        "Neurological outcomes in treated migraine",
        "https://publisher.example/final",
        abstract=abstract,
        doi="10.1000/shared",
        status="enriched",
    )

    assert deduplicate_enriched([without_doi, with_doi]) == [with_doi]


def test_deduplicates_high_confidence_near_identical_content():
    first = _enriched(
        "A clinical migraine analysis",
        "https://one.example/paper",
        abstract=(
            "The clinical analysis enrolled adults with episodic migraine and measured "
            "headache frequency, symptom severity, functional impairment, medication use, "
            "and treatment safety throughout twelve weeks of structured follow up."
        ),
    )
    second = _enriched(
        "Outcomes following migraine treatment",
        "https://two.example/paper",
        abstract=(
            "The clinical analysis enrolled adults with episodic migraine and measured "
            "headache frequency, symptom severity, functional impairment, medicine use, "
            "and treatment safety throughout twelve weeks of structured follow-up."
        ),
    )

    assert len(deduplicate_enriched([first, second])) == 1


def test_does_not_merge_distinct_results_with_generic_short_descriptions():
    first = _enriched(
        "Migraine treatment A",
        "https://one.example/a",
        snippet="A study of migraine treatment safety and clinical outcomes.",
    )
    second = _enriched(
        "Migraine treatment B",
        "https://two.example/b",
        snippet="A study of migraine treatment safety and clinical outcomes.",
    )

    assert deduplicate_enriched([first, second]) == [first, second]
