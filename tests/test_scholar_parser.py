from __future__ import annotations

import pytest

from newsletter_agent.infrastructure.scholar_parser import (
    ScholarAlertParseError,
    canonicalize_url,
    parse_scholar_alert,
    unwrap_google_redirect,
)


def test_parses_scholar_style_html_and_ignores_unsubscribe():
    html = """
    <div>
      <h3><a href="https://example.org/article?utm_source=scholar">A meaningful paper title</a></h3>
      <div>A. Author - Example Journal, 2026</div>
      <div>This study evaluates a useful research question.</div>
    </div>
    <a href="https://scholar.google.com/scholar_alerts?view_op=cancel_alert">Cancel alert</a>
    """
    items = parse_scholar_alert(html=html, text="", message_id="gmail-1")
    assert len(items) == 1
    assert items[0].title == "A meaningful paper title"
    assert items[0].url == "https://example.org/article"
    assert items[0].source_message_id == "gmail-1"


def test_unwraps_google_redirect_and_removes_tracking():
    wrapped = "https://www.google.com/url?q=https%3A%2F%2Fexample.org%2Fpaper%3Futm_medium%3Demail%26id%3D7"
    assert unwrap_google_redirect(wrapped).startswith("https://example.org/paper")
    assert canonicalize_url(wrapped) == "https://example.org/paper?id=7"


def test_parses_google_scholar_url_redirect():
    html = """
    <h3>
      <a href="https://scholar.google.com/scholar_url?url=https%3A%2F%2Fexample.org%2Fpaper%3Futm_source%3Dalert%26id%3D9&amp;hl=en&amp;sa=X">
        A Scholar alert paper reached through its redirect
      </a>
    </h3>
    <div>A. Author - Example Journal, 2026</div>
    """

    items = parse_scholar_alert(html=html, text="", message_id="gmail-redirect")

    assert len(items) == 1
    assert items[0].url == "https://example.org/paper?id=9"


def test_unparseable_labeled_message_is_fatal():
    with pytest.raises(ScholarAlertParseError):
        parse_scholar_alert(html="<p>No results here</p>", text="", message_id="gmail-2")
