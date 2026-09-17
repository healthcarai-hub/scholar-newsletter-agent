from __future__ import annotations

from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

from newsletter_agent.config import AppConfig
from newsletter_agent.domain.models import (
    ClassifiedItem,
    EnrichedItem,
    NewsletterIssue,
    ResearchItem,
    RunContext,
)
from newsletter_agent.infrastructure.rendering import JinjaNewsletterRenderer


def _issue(profile, items):
    cutoff = datetime(2026, 9, 3, 20, 0, tzinfo=ZoneInfo("Asia/Kolkata"))
    context = RunContext(
        workspace_id="default",
        profile_id="migraine",
        topic=profile.topic,
        window_start=cutoff - timedelta(days=7),
        cutoff=cutoff,
        issue_key="default:migraine:2026-09-03",
    )
    return NewsletterIssue(
        context=context,
        subject="This Week in migraine research",
        title="Weekly migraine research Digest",
        summary="A concise overview of the available research.",
        items=tuple(items),
        recipients=("editor@example.com",),
        accent_color="#155eef",
        subtitle="Curated research",
        footer="For review.",
    )


def _entry(
    category="Mechanisms",
    source_label=None,
    priority="normal",
    source_type="research_paper",
    url="https://example.org/paper",
):
    original = ResearchItem(
        title="Paper <unsafe>",
        url=url,
        snippet="Description",
        source_message_id="message-1",
    )
    enriched = EnrichedItem(
        item=original,
        canonical_url=original.url,
        fingerprint="fingerprint",
        source_text="Title: Paper",
    )
    return ClassifiedItem(
        enriched=enriched,
        category=category,
        headline="Useful <headline>",
        brief="Grounded description.",
        source_label=source_label,
        priority=priority,
        source_type=source_type,
    )


def test_empty_categories_are_absent_from_html_and_text(config_dict):
    config = AppConfig.model_validate(config_dict)
    profile = config.profiles["migraine"]
    rendered = JinjaNewsletterRenderer().render(_issue(profile, [_entry()]), profile)
    assert "Mechanisms" in rendered.html
    assert "Treatments" not in rendered.html
    assert "TREATMENTS" not in rendered.text
    assert "display:flex" not in rendered.html
    assert "September 4, 2026" in rendered.html
    assert "September 4, 2026" in rendered.text


def test_categories_are_ordered_by_descending_result_count(config_dict):
    config = AppConfig.model_validate(config_dict)
    profile = config.profiles["migraine"]
    rendered = JinjaNewsletterRenderer().render(
        _issue(
            profile,
            [
                _entry("Mechanisms"),
                _entry("Treatments"),
                _entry("Treatments"),
            ],
        ),
        profile,
    )
    assert rendered.html.index(">Treatments</a>") < rendered.html.index(">Mechanisms</a>")
    assert rendered.text.index("TREATMENTS") < rendered.text.index("MECHANISMS")


def test_equal_category_counts_retain_configured_order(config_dict):
    config = AppConfig.model_validate(config_dict)
    profile = config.profiles["migraine"]
    rendered = JinjaNewsletterRenderer().render(
        _issue(profile, [_entry("Treatments"), _entry("Mechanisms")]), profile
    )
    assert rendered.html.index(">Mechanisms</a>") < rendered.html.index(">Treatments</a>")
    assert rendered.text.index("MECHANISMS") < rendered.text.index("TREATMENTS")


def test_categories_with_research_precede_larger_low_priority_only_categories(config_dict):
    config = AppConfig.model_validate(config_dict)
    profile = config.profiles["migraine"]
    rendered = JinjaNewsletterRenderer().render(
        _issue(
            profile,
            [
                _entry("Mechanisms", priority="low"),
                _entry("Mechanisms", priority="low"),
                _entry("Treatments", priority="normal"),
            ],
        ),
        profile,
    )

    assert rendered.html.index(">Treatments</a>") < rendered.html.index(">Mechanisms</a>")
    assert rendered.text.index("TREATMENTS") < rendered.text.index("MECHANISMS")


def test_llm_and_source_text_are_html_escaped(config_dict):
    config = AppConfig.model_validate(config_dict)
    profile = config.profiles["migraine"]
    rendered = JinjaNewsletterRenderer().render(_issue(profile, [_entry()]), profile)
    assert "Useful &lt;headline&gt;" in rendered.html
    assert "Useful <headline>" not in rendered.html
    assert "Useful <headline>" in rendered.text
    assert "&lt;headline&gt;" not in rendered.text


def test_bibliographic_metadata_line_is_not_rendered(config_dict):
    config = AppConfig.model_validate(config_dict)
    profile = config.profiles["migraine"]
    metadata_line = "Journal of Examples · 2026-09-04 · DOI 10.1000/example"
    rendered = JinjaNewsletterRenderer().render(
        _issue(profile, [_entry(source_label=metadata_line)]), profile
    )
    assert metadata_line not in rendered.html
    assert metadata_line not in rendered.text


def test_compact_platform_and_reader_friendly_source_type_are_rendered(config_dict):
    config = AppConfig.model_validate(config_dict)
    profile = config.profiles["migraine"]
    rendered = JinjaNewsletterRenderer().render(
        _issue(
            profile,
            [
                _entry(
                    source_type="seo_content",
                    priority="low",
                    url="https://noobru.com/blogs/articles/example",
                )
            ],
        ),
        profile,
    )

    assert "NooBru" in rendered.html
    assert "Consumer Editorial" in rendered.html
    assert "font-size:11px" in rendered.html
    assert "NooBru · Consumer Editorial" in rendered.text
    assert "SEO" not in rendered.html


def test_zero_result_issue_has_no_category_sections(config_dict):
    config = AppConfig.model_validate(config_dict)
    profile = config.profiles["migraine"]
    rendered = JinjaNewsletterRenderer().render(_issue(profile, []), profile)
    assert "No new alert results" in rendered.html
    assert "Mechanisms" not in rendered.html
    assert "Treatments" not in rendered.html


def test_fallback_category_uses_editorial_label(config_dict):
    config = AppConfig.model_validate(config_dict)
    profile = config.profiles["migraine"]
    rendered = JinjaNewsletterRenderer().render(
        _issue(profile, [_entry("Additional Research")]), profile
    )
    assert "Additional Research" in rendered.html
    assert "ADDITIONAL RESEARCH" in rendered.text
    assert "Other / Uncategorized" not in rendered.html


def test_configurable_intro_closing_and_signature_are_rendered(config_dict):
    newsletter = config_dict["profiles"]["migraine"]["newsletter"]
    newsletter.update(
        {
            "greeting": "Good morning Curt,",
            "introduction": "Here is the weekly research wrap-up.",
            "results_lead": "Here are the latest publications:",
            "closing": "Have a great weekend.",
            "signature": {
                "name": "NICOLE MELAMED",
                "title": "PA",
                "email": "nicole@healthcar.co",
                "phone": "763-258-7678",
                "disclaimer": "Confidential.",
            },
        }
    )
    config = AppConfig.model_validate(config_dict)
    profile = config.profiles["migraine"]

    rendered = JinjaNewsletterRenderer().render(_issue(profile, [_entry()]), profile)

    for expected in (
        "Good morning Curt,",
        "Here is the weekly research wrap-up.",
        "Here are the latest publications:",
        "Have a great weekend.",
        "NICOLE MELAMED",
        "nicole@healthcar.co",
    ):
        assert expected in rendered.html
        assert expected in rendered.text

    assert rendered.html.index("Good morning Curt,") < rendered.html.index(
        "This week at a glance"
    )
    assert rendered.text.index("Good morning Curt,") < rendered.text.index(
        "A concise overview"
    )
