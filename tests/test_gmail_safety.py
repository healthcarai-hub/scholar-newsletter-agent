import base64
from datetime import datetime, timedelta, timezone

import pytest

from newsletter_agent.application.pipeline import run_context
from newsletter_agent.config import AppConfig
from newsletter_agent.infrastructure.gmail import (
    GmailAdapter,
    gmail_window_query,
    message_is_in_run_window,
)
from newsletter_agent.infrastructure.scholar_parser import ScholarAlertParseError


def test_gmail_adapter_exposes_no_send_method():
    assert not hasattr(GmailAdapter, "send")


def test_gmail_adapter_exposes_no_label_mutation_method():
    assert not hasattr(GmailAdapter, "acknowledge")


class _Executable:
    def __init__(self, result):
        self.result = result

    def execute(self):
        return self.result


class _FailingExecutable:
    def execute(self):
        raise RuntimeError("invalid_scope: Bad Request")


class _Labels:
    def __init__(self, result):
        self.result = result

    def list(self, **kwargs):
        assert kwargs == {"userId": "me"}
        return _Executable(self.result)


class _FailingLabels:
    def list(self, **kwargs):
        assert kwargs == {"userId": "me"}
        return _FailingExecutable()


class _FailingUsers:
    def labels(self):
        return _FailingLabels()


class _FailingService:
    def users(self):
        return _FailingUsers()


class _Users:
    def __init__(self, result):
        self.result = result

    def labels(self):
        return _Labels(self.result)


class _Service:
    def __init__(self, result):
        self.result = result

    def users(self):
        return _Users(self.result)


def test_gmail_label_id_is_resolved_from_configured_name(config_dict):
    profile = AppConfig.model_validate(config_dict).profiles["migraine"]
    adapter = GmailAdapter(
        _Service(
            {
                "labels": [
                    {"id": "Label_7", "name": "scholar alerts/migraine"},
                    {"id": "INBOX", "name": "INBOX"},
                ]
            }
        )
    )
    assert adapter._resolve_label_sync(profile) == "Label_7"


def test_missing_gmail_label_name_fails_safely(config_dict):
    profile = AppConfig.model_validate(config_dict).profiles["migraine"]
    adapter = GmailAdapter(_Service({"labels": [{"id": "INBOX", "name": "INBOX"}]}))
    with pytest.raises(RuntimeError, match="Gmail label not found"):
        adapter._resolve_label_sync(profile)


def test_invalid_scope_has_actionable_reauthorization_message(config_dict):
    profile = AppConfig.model_validate(config_dict).profiles["migraine"]
    adapter = GmailAdapter(_FailingService())

    with pytest.raises(RuntimeError, match="newsletter-agent auth"):
        adapter._resolve_label_sync(profile)


def test_only_messages_inside_the_seven_day_window_are_selected():
    cutoff = datetime(2026, 9, 3, 14, 30, tzinfo=timezone.utc)
    context = run_context(
        workspace_id="default",
        profile_id="migraine",
        topic="migraine research",
        cutoff=cutoff,
        lookback_days=7,
    )

    assert not message_is_in_run_window(context, cutoff - timedelta(days=7))
    assert message_is_in_run_window(
        context, cutoff - timedelta(days=7) + timedelta(seconds=1)
    )
    assert message_is_in_run_window(context, cutoff)
    assert not message_is_in_run_window(context, cutoff + timedelta(seconds=1))


def test_gmail_query_limits_candidates_to_the_run_window():
    cutoff = datetime(2026, 9, 3, 14, 30, tzinfo=timezone.utc)
    context = run_context(
        workspace_id="default",
        profile_id="migraine",
        topic="migraine research",
        cutoff=cutoff,
        lookback_days=7,
    )

    assert gmail_window_query(context) == (
        f"after:{int((cutoff - timedelta(days=7)).timestamp())} "
        f"before:{int(cutoff.timestamp()) + 1}"
    )


# --- per-message parse tolerance -------------------------------------------------

_CUTOFF = datetime(2026, 9, 3, 14, 30, tzinfo=timezone.utc)
_GOOD_HTML = (
    '<html><body><h3><a href="https://example.org/paper-one">'
    "A study of migraine mechanisms</a></h3></body></html>"
)
_BAD_HTML = "<html><body><p>Your alert was created. Nothing to see here.</p></body></html>"


def _b64(value: str) -> str:
    return base64.urlsafe_b64encode(value.encode()).decode()


def _gmail_message(message_id: str, html: str) -> dict:
    received = _CUTOFF - timedelta(days=1)
    return {
        "id": message_id,
        "threadId": f"t-{message_id}",
        "internalDate": str(int(received.timestamp() * 1000)),
        "payload": {
            "mimeType": "text/html",
            "headers": [{"name": "Subject", "value": f"Alert {message_id}"}],
            "body": {"data": _b64(html)},
        },
    }


class _ListingService:
    """Fake Gmail service: one label, a message list, and full message bodies."""

    def __init__(self, messages: dict[str, dict]):
        self._messages = messages

    def users(self):
        return self

    def labels(self):
        return self

    def messages(self):
        return self

    def list(self, **kwargs):
        if "labelIds" in kwargs:
            return _Executable({"messages": [{"id": mid} for mid in self._messages]})
        return _Executable({"labels": [{"id": "Label_7", "name": "scholar alerts/migraine"}]})

    def get(self, *, userId, id, format):
        return _Executable(self._messages[id])


def _pending(config_dict, messages: dict[str, dict]):
    profile = AppConfig.model_validate(config_dict).profiles["migraine"]
    context = run_context(
        workspace_id="default",
        profile_id="migraine",
        topic="migraine research",
        cutoff=_CUTOFF,
        lookback_days=7,
    )
    return GmailAdapter(_ListingService(messages))._list_pending_sync(context, profile)


def test_unparseable_message_is_skipped_and_valid_alerts_still_processed(
    config_dict, caplog
):
    messages = {
        "bad1": _gmail_message("bad1", _BAD_HTML),
        "good1": _gmail_message("good1", _GOOD_HTML),
    }

    with caplog.at_level("WARNING", logger="newsletter_agent.infrastructure.gmail"):
        batch = _pending(config_dict, messages)

    assert [m.message_id for m in batch.messages] == ["good1"]
    assert len(batch.items) == 1
    assert "gmail_message_skipped" in caplog.text
    assert "bad1" in caplog.text


def test_skipped_message_is_not_included_in_the_batch(config_dict):
    batch = _pending(
        config_dict,
        {
            "bad1": _gmail_message("bad1", _BAD_HTML),
            "good1": _gmail_message("good1", _GOOD_HTML),
        },
    )

    # Skipped messages must not be treated as consumed source messages.
    assert "bad1" not in {m.message_id for m in batch.messages}


def test_all_messages_unparseable_still_fails_the_profile(config_dict):
    with pytest.raises(ScholarAlertParseError, match="bad1.*bad2"):
        _pending(
            config_dict,
            {
                "bad1": _gmail_message("bad1", _BAD_HTML),
                "bad2": _gmail_message("bad2", _BAD_HTML),
            },
        )


def test_no_messages_in_window_still_returns_an_empty_batch(config_dict):
    batch = _pending(config_dict, {})

    assert batch.messages == ()
    assert batch.items == ()
