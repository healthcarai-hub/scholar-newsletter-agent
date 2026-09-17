from __future__ import annotations

from newsletter_agent.config import ContentPolicyConfig
from newsletter_agent.domain.models import ClassifiedItem, EnrichedItem, ResearchItem
from newsletter_agent.infrastructure.source_quality import (
    apply_content_policy,
    infer_source_type,
    with_source_quality,
)


def _item(title: str, url: str, snippet: str, *, doi: str | None = None) -> EnrichedItem:
    research = ResearchItem(
        title=title,
        url=url,
        snippet=snippet,
        source_message_id="message-1",
        doi=doi,
    )
    return EnrichedItem(
        item=research,
        canonical_url=url,
        fingerprint=url,
        source_text=f"Title: {title}\nScholar snippet: {snippet}",
        doi=doi,
        link_status="available",
    )


def _classified(title: str, url: str, snippet: str) -> ClassifiedItem:
    base = ClassifiedItem(
        enriched=_item(title, url, snippet),
        category="Cognition",
        headline=title,
        brief=snippet,
    )
    return with_source_quality(base, "seo_content")


def test_academic_identifiers_override_seo_styling():
    item = _item(
        "Best evidence from a systematic review",
        "https://www.sciencedirect.com/science/article/pii/example",
        "A systematic review of controlled studies.",
        doi="10.1000/example",
    )

    assert infer_source_type(item, "seo_content") == "academic_review"


def test_commercial_blog_is_low_priority():
    item = _classified(
        "Best Nootropics for Focus in 2026: Ranked by Evidence",
        "https://noobru.com/blogs/articles/best-nootropics-focus",
        "A commercial guide compares nootropic products for focus.",
    )

    assert item.source_type in {"seo_content", "marketing_page"}
    assert item.priority == "low"


def test_commercial_url_slug_is_evidence_when_alert_title_is_generic():
    item = _item(
        "Thirty years of real-world testing",
        "https://revgear.com/blogs/blog/transparent-labs-bulk-pre-workout-review",
        "A brand-focused article.",
    )

    assert infer_source_type(item, "research_news") == "marketing_page"


def test_similar_low_priority_pages_on_same_domain_keep_one_representative():
    first = _classified(
        "Best Nootropics for Focus in 2026 Ranked by Evidence",
        "https://noobru.com/blogs/articles/best-nootropics-focus",
        "This guide ranks nootropic ingredients for focus using evidence and product claims.",
    )
    second = _classified(
        "Nootropics for Focus: 7 Ingredients Ranked by Evidence",
        "https://noobru.com/blogs/articles/nootropics-focus-seven",
        "This guide ranks seven nootropic ingredients for focus using evidence and product claims.",
    )

    retained = apply_content_policy([first, second], ContentPolicyConfig())

    assert len(retained) == 1


def test_same_publisher_does_not_cluster_distinct_academic_papers():
    first = with_source_quality(
        ClassifiedItem(
            enriched=_item(
                "Migraine prevention trial",
                "https://www.sciencedirect.com/science/article/pii/one",
                "A randomized prevention trial.",
            ),
            category="Treatment",
            headline="Migraine prevention trial",
            brief="A randomized prevention trial.",
        ),
        "research_paper",
    )
    second = with_source_quality(
        ClassifiedItem(
            enriched=_item(
                "Migraine biomarker analysis",
                "https://www.sciencedirect.com/science/article/pii/two",
                "An analysis of a diagnostic biomarker.",
            ),
            category="Diagnostics",
            headline="Migraine biomarker analysis",
            brief="An analysis of a diagnostic biomarker.",
        ),
        "research_paper",
    )

    assert apply_content_policy([first, second], ContentPolicyConfig()) == [first, second]


def test_low_priority_items_render_after_research_items():
    low = _classified(
        "Best Nootropics for Focus",
        "https://noobru.com/blogs/articles/best-nootropics-focus",
        "A commercial guide.",
    )
    research = with_source_quality(
        ClassifiedItem(
            enriched=_item(
                "L-theanine clinical study",
                "https://pubmed.ncbi.nlm.nih.gov/123",
                "A controlled study.",
            ),
            category="Cognition",
            headline="L-theanine clinical study",
            brief="A controlled study.",
        ),
        "research_paper",
    )

    assert apply_content_policy([low, research], ContentPolicyConfig()) == [research, low]


def test_only_one_low_priority_item_is_kept_per_domain_across_distinct_clusters():
    focus = _classified(
        "Best Nootropics for Focus",
        "https://noobru.com/blogs/articles/best-nootropics-focus",
        "A commercial guide about focus products.",
    )
    sleep = _classified(
        "A Complete Guide to Sleep Supplements",
        "https://noobru.com/blogs/articles/sleep-supplements-guide",
        "A different commercial guide about products for sleep.",
    )

    retained = apply_content_policy([focus, sleep], ContentPolicyConfig())

    assert len(retained) == 1


def test_commercial_domain_cap_includes_ambiguous_news_from_same_platform():
    commercial = _classified(
        "Best Nootropics for Focus",
        "https://noobru.com/blogs/articles/best-nootropics-focus",
        "A commercial guide about focus products.",
    )
    ambiguous = with_source_quality(
        ClassifiedItem(
            enriched=_item(
                "How Nootropics Support Focus and Mental Clarity",
                "https://noobru.com/blogs/articles/how-nootropics-support-focus",
                "An article discussing ingredients used for focus.",
            ),
            category="Cognition",
            headline="How Nootropics Support Focus and Mental Clarity",
            brief="An article discussing ingredients used for focus.",
        ),
        "research_news",
    )
    assert ambiguous.priority == "normal"

    retained = apply_content_policy([commercial, ambiguous], ContentPolicyConfig())

    assert len(retained) == 1
    assert retained[0].priority == "low"
