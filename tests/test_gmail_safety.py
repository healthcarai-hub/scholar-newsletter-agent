from datetime import datetime, timedelta, timezone

import pytest

from newsletter_agent.application.pipeline import run_context
from newsletter_agent.config import AppConfig
from newsletter_agent.infrastructure.gmail import GmailAdapter, message_is_in_run_window


def test_gmail_adapter_exposes_no_send_method():
    assert not hasattr(GmailAdapter, "send")


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
