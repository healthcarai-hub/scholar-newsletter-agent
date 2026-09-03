from __future__ import annotations

from collections import OrderedDict
from pathlib import Path

from jinja2 import Environment, FileSystemLoader

from newsletter_agent.config import ProfileConfig
from newsletter_agent.domain.models import NewsletterIssue, RenderedNewsletter
from newsletter_agent.domain.scheduling import publication_date


class JinjaNewsletterRenderer:
    def __init__(self, template_directory: str | Path | None = None) -> None:
        directory = Path(template_directory) if template_directory else Path(__file__).parents[1] / "templates"
        self.environment = Environment(
            loader=FileSystemLoader(directory),
            # Only the HTML alternative should entity-escape content. Applying
            # autoescape to every .j2 template leaks entities such as &amp; into
            # the plain-text email alternative.
            autoescape=lambda template_name: bool(
                template_name and template_name.endswith(".html.j2")
            ),
            trim_blocks=True,
            lstrip_blocks=True,
        )

    def render(self, issue: NewsletterIssue, profile: ProfileConfig) -> RenderedNewsletter:
        grouped: OrderedDict[str, list] = OrderedDict(
            (category, []) for category in profile.allowed_categories
        )
        for item in issue.items:
            grouped.setdefault(item.category, []).append(item)

        # This filtered, stably sorted collection is the single source for both
        # navigation and sections. Empty categories cannot leak into either
        # representation, and equal counts retain the profile's configured order.
        populated = [(name, items) for name, items in grouped.items() if items]
        populated.sort(key=lambda pair: len(pair[1]), reverse=True)
        sections = [
            {"name": name, "anchor": f"category-{index + 1}", "items": items}
            for index, (name, items) in enumerate(populated)
        ]
        context = {
            "issue": issue,
            "newsletter": profile.newsletter,
            "sections": sections,
            "result_count": len(issue.items),
            "issue_date": publication_date(issue.context.cutoff).strftime("%B %-d, %Y"),
        }
        html = self.environment.get_template("newsletter.html.j2").render(**context)
        text = self.environment.get_template("newsletter.txt.j2").render(**context).strip() + "\n"
        return RenderedNewsletter(
            subject=issue.subject,
            html=html,
            text=text,
            recipients=issue.recipients,
        )
