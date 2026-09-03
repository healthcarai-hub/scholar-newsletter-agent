from __future__ import annotations

from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

from newsletter_agent.config import AppConfig
from newsletter_agent.domain.models import (
    ClassifiedItem,
    EnrichedItem,
    NewsletterIssue,
    ResearchItem,
)
from newsletter_agent.infrastructure.rendering import JinjaNewsletterRenderer
from newsletter_agent.application.pipeline import _template_values, run_context


def render_sample_previews(config: AppConfig, output_directory: str | Path) -> list[Path]:
    output = Path(output_directory)
    output.mkdir(parents=True, exist_ok=True)
    cutoff = datetime(2026, 9, 3, 20, 0, tzinfo=ZoneInfo(config.runtime.timezone))
    renderer = JinjaNewsletterRenderer()
    written: list[Path] = []
    for profile_id, profile in config.profiles.items():
        context = run_context(
            workspace_id=config.runtime.workspace_id,
            profile_id=profile_id,
            topic=profile.topic,
            cutoff=cutoff,
        )
        original = ResearchItem(
            title=f"A representative advance in {profile.topic}",
            url="https://example.org/research/sample",
            snippet="A sample alert description used only to preview the newsletter layout.",
            source_message_id="sample-message",
            authors="A. Researcher, B. Scientist",
            venue="Journal of Sample Research",
            published_at="2026",
            doi="10.1000/example",
        )
        enriched = EnrichedItem(
            item=original,
            canonical_url=original.url,
            fingerprint="sample-fingerprint",
            source_text=f"Title: {original.title}\nScholar snippet: {original.snippet}",
            authors=original.authors,
            venue=original.venue,
            published_at=original.published_at,
            doi=original.doi,
            enrichment_status="enriched",
        )
        entry = ClassifiedItem(
            enriched=enriched,
            category=profile.categories[0].name,
            headline=f"Representative evidence update in {profile.topic}",
            brief=(
                "This sample card demonstrates the intended hierarchy: an informative linked "
                "headline and a concise grounded description designed for fast review."
            ),
        )
        values = _template_values(context, profile_id, 1)
        issue = NewsletterIssue(
            context=context,
            subject=profile.newsletter.subject_template.format_map(values),
            title=profile.newsletter.title_template.format_map(values),
            summary=(
                f"This week's {profile.topic} digest highlights a representative publication. "
                "The layout prioritizes a quick editorial overview, consistent evidence cards, "
                "and clear source links. Categories without results are omitted automatically."
            ),
            items=(entry,),
            recipients=profile.newsletter.recipients,
            accent_color=profile.newsletter.accent_color,
            subtitle=profile.newsletter.subtitle,
            footer=profile.newsletter.footer,
        )
        rendered = renderer.render(issue, profile)
        html_path = output / f"sample-{profile_id}.html"
        text_path = output / f"sample-{profile_id}.txt"
        html_path.write_text(rendered.html, encoding="utf-8")
        text_path.write_text(rendered.text, encoding="utf-8")
        written.extend([html_path, text_path])
    return written
