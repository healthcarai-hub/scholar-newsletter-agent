from __future__ import annotations

import asyncio
from datetime import datetime, timezone

import pytest

from newsletter_agent.application.pipeline import (
    NewsletterPipeline,
    _template_values,
    run_context,
    scheduled_cutoff,
)
from newsletter_agent.config import AppConfig
from newsletter_agent.domain.models import (
    AlertBatch,
    ClassifiedItem,
    EnrichedItem,
    PersistedIssue,
    RenderedNewsletter,
    ResearchItem,
    SourceMessage,
)


class FakeGmail:
    def __init__(self, batch, fail_draft=False):
        self.batch = batch
        self.fail_draft = fail_draft
        self.created = []
        self.contexts = []

    async def list_pending(self, context, profile):
        self.contexts.append(context)
        return self.batch

    async def create_or_find(self, rendered, issue_key, existing_draft_id=None):
        if self.fail_draft:
            raise RuntimeError("draft failed")
        self.created.append(issue_key)
        return existing_draft_id or "draft-1"


class FakeEnricher:
    async def enrich(self, item):
        return EnrichedItem(
            item=item,
            canonical_url=item.url,
            fingerprint=item.title.casefold(),
            source_text=f"Title: {item.title}\nScholar snippet: {item.snippet}",
        )


class FakeCategorizer:
    async def classify(self, item, profile):
        if profile.topic == "broken topic":
            raise RuntimeError("model failure")
        return ClassifiedItem(
            enriched=item,
            category=profile.categories[0].name,
            headline=item.item.title,
            brief=item.item.snippet,
        )

    async def summarize_issue(self, items, profile):
        return "A grounded weekly overview." if items else f"No new {profile.topic} results."


class CountingCategorizer(FakeCategorizer):
    def __init__(self):
        self.classify_calls = 0

    async def classify(self, item, profile):
        self.classify_calls += 1
        return await super().classify(item, profile)


class FakeRenderer:
    def render(self, issue, profile):
        return RenderedNewsletter(issue.subject, "<html></html>", "text", issue.recipients)


class FakeEmbedder:
    provider_name = "fake"
    model_name = "fake-model"

    async def embed(self, texts):
        return [[1.0, 0.0] for _ in texts]


class FailingEmbedder(FakeEmbedder):
    async def embed(self, texts):
        raise RuntimeError("embedding service unavailable")


class FakeRepository:
    def __init__(self, draft_id=None):
        self.draft_id = draft_id
        self.failed = []
        self.completed = []
        self.persisted = []
        self.saved_vectors = []
        self.embedding_failures = []

    async def begin_or_resume(self, context, profile, batch):
        return PersistedIssue("issue-1", self.draft_id, "processing", "lease")

    async def persist_item(self, context, issue_id, item, position, provider, model):
        self.persisted.append(item)
        return "chunk-1", "embedding-1", True

    async def save_embedding(self, embedding_id, vector):
        self.saved_vectors.append((embedding_id, vector))

    async def mark_embedding_failed(self, embedding_id, error):
        self.embedding_failures.append((embedding_id, error))

    async def record_draft(self, issue_id, draft_id, subject, summary):
        self.draft_id = draft_id

    async def complete_issue(self, issue_id):
        self.completed.append(issue_id)

    async def fail_issue(self, issue_id, error):
        self.failed.append((issue_id, error))

    async def list_pending_embeddings(self, limit):
        return []


def _batch():
    message = SourceMessage("message-1", None, datetime.now(timezone.utc), "Scholar Alert")
    item = ResearchItem(
        title="A useful study",
        url="https://example.org/paper",
        snippet="A grounded description.",
        source_message_id=message.message_id,
    )
    return AlertBatch((message,), (item,), "label-1")


def _pipeline(config, gmail, repository):
    return NewsletterPipeline(
        config=config,
        alert_source=gmail,
        enricher=FakeEnricher(),
        categorizer=FakeCategorizer(),
        renderer=FakeRenderer(),
        publisher=gmail,
        embedding_provider=FakeEmbedder(),
        repository=repository,
    )


def _pipeline_with_embedder(config, gmail, repository, embedder):
    pipeline = _pipeline(config, gmail, repository)
    pipeline.embedding_provider = embedder
    return pipeline


def test_normal_run_creates_draft_without_mutating_gmail_labels(config_dict):
    config = AppConfig.model_validate(config_dict)
    gmail = FakeGmail(_batch())
    repository = FakeRepository()
    results = asyncio.run(
        _pipeline(config, gmail, repository).run(
            now=datetime(2026, 9, 3, 21, 0, tzinfo=timezone.utc)
        )
    )
    assert results[0].state == "completed"
    assert gmail.created
    assert repository.saved_vectors


def test_keep_labels_creates_draft_without_acknowledging_messages(config_dict):
    config = AppConfig.model_validate(config_dict)
    gmail = FakeGmail(_batch())
    repository = FakeRepository()
    results = asyncio.run(
        _pipeline(config, gmail, repository).run(
            now=datetime(2026, 9, 3, 21, 0, tzinfo=timezone.utc),
            keep_labels=True,
        )
    )
    assert results[0].state == "completed_labels_kept"
    assert gmail.created == ["default:migraine:2026-09-03:labels-kept:7d"]
    assert repository.completed == ["issue-1"]


def test_one_run_lookback_override_is_isolated_in_inspection_issue_key(config_dict):
    config = AppConfig.model_validate(config_dict)
    gmail = FakeGmail(_batch())
    repository = FakeRepository()
    results = asyncio.run(
        _pipeline(config, gmail, repository).run(
            now=datetime(2026, 9, 3, 21, 0, tzinfo=timezone.utc),
            keep_labels=True,
            lookback_days=10,
        )
    )
    assert results[0].state == "completed_labels_kept"
    assert gmail.created == ["default:migraine:2026-09-03:labels-kept:10d"]
    assert (gmail.contexts[0].cutoff - gmail.contexts[0].window_start).days == 10


def test_lookback_override_rejects_out_of_range_value(config_dict):
    config = AppConfig.model_validate(config_dict)
    with pytest.raises(ValueError, match="between 1 and 90"):
        asyncio.run(
            _pipeline(config, FakeGmail(_batch()), FakeRepository()).run(
                keep_labels=True,
                lookback_days=91,
            )
        )


def test_normal_run_keeps_production_issue_key(config_dict):
    config = AppConfig.model_validate(config_dict)
    gmail = FakeGmail(_batch())
    repository = FakeRepository()
    asyncio.run(
        _pipeline(config, gmail, repository).run(
            now=datetime(2026, 9, 3, 21, 0, tzinfo=timezone.utc)
        )
    )
    assert gmail.created == ["default:migraine:2026-09-03"]
    assert repository.completed == ["issue-1"]


def test_keep_labels_is_honored_when_recovering_existing_draft(config_dict):
    config = AppConfig.model_validate(config_dict)
    gmail = FakeGmail(_batch())
    repository = FakeRepository(draft_id="existing-draft")
    results = asyncio.run(
        _pipeline(config, gmail, repository).run(
            now=datetime(2026, 9, 3, 21, 0, tzinfo=timezone.utc),
            keep_labels=True,
        )
    )
    assert results[0].state == "recovered_labels_kept"
    assert gmail.created == []


def test_draft_failure_preserves_label(config_dict):
    config = AppConfig.model_validate(config_dict)
    gmail = FakeGmail(_batch(), fail_draft=True)
    repository = FakeRepository()
    results = asyncio.run(
        _pipeline(config, gmail, repository).run(
            now=datetime(2026, 9, 3, 21, 0, tzinfo=timezone.utc)
        )
    )
    assert results[0].state == "failed"
    assert repository.failed


def test_confirmed_dead_links_are_removed_before_categorization(config_dict):
    config = AppConfig.model_validate(config_dict)
    gmail = FakeGmail(_batch())
    repository = FakeRepository()
    categorizer = CountingCategorizer()

    class DeadLinkEnricher(FakeEnricher):
        async def enrich(self, item):
            enriched = await super().enrich(item)
            return EnrichedItem(
                item=enriched.item,
                canonical_url=enriched.canonical_url,
                fingerprint=enriched.fingerprint,
                source_text=enriched.source_text,
                link_status="dead",
                http_status=404,
            )

    pipeline = NewsletterPipeline(
        config=config,
        alert_source=gmail,
        enricher=DeadLinkEnricher(),
        categorizer=categorizer,
        renderer=FakeRenderer(),
        publisher=gmail,
        embedding_provider=FakeEmbedder(),
        repository=repository,
    )
    results = asyncio.run(
        pipeline.run(now=datetime(2026, 9, 3, 21, 0, tzinfo=timezone.utc))
    )

    assert results[0].state == "completed"
    assert results[0].result_count == 0
    assert categorizer.classify_calls == 0
    assert repository.persisted == []


def test_existing_draft_recovers_without_recreating(config_dict):
    config = AppConfig.model_validate(config_dict)
    gmail = FakeGmail(_batch())
    repository = FakeRepository(draft_id="existing-draft")
    results = asyncio.run(
        _pipeline(config, gmail, repository).run(
            now=datetime(2026, 9, 3, 21, 0, tzinfo=timezone.utc)
        )
    )
    assert results[0].state == "recovered"
    assert gmail.created == []


def test_embedding_failure_does_not_block_draft(config_dict):
    config = AppConfig.model_validate(config_dict)
    gmail = FakeGmail(_batch())
    repository = FakeRepository()
    results = asyncio.run(
        _pipeline_with_embedder(config, gmail, repository, FailingEmbedder()).run(
            now=datetime(2026, 9, 3, 21, 0, tzinfo=timezone.utc)
        )
    )
    assert results[0].state == "completed"
    assert repository.embedding_failures
    assert gmail.created


def test_profile_can_skip_rag_indexing_without_blocking_draft(config_dict):
    config_dict["profiles"]["migraine"]["rag_indexing_enabled"] = False
    config = AppConfig.model_validate(config_dict)
    gmail = FakeGmail(_batch())
    repository = FakeRepository()

    results = asyncio.run(
        _pipeline(config, gmail, repository).run(
            now=datetime(2026, 9, 3, 21, 0, tzinfo=timezone.utc)
        )
    )

    assert results[0].state == "completed"
    assert repository.persisted == []
    assert repository.saved_vectors == []
    assert gmail.created


def test_profile_failure_is_isolated(config_dict):
    config_dict["profiles"]["broken"] = {
        **config_dict["profiles"]["migraine"],
        "topic": "broken topic",
        "gmail": {"label_name": "Scholar Alerts/Broken"},
    }
    config = AppConfig.model_validate(config_dict)
    gmail = FakeGmail(_batch())
    repository = FakeRepository()
    results = asyncio.run(
        _pipeline(config, gmail, repository).run(
            now=datetime(2026, 9, 3, 21, 0, tzinfo=timezone.utc)
        )
    )
    states = {result.profile_id: result.state for result in results}
    assert states == {"migraine": "completed", "broken": "failed"}


def test_weekly_cutoff_is_fixed_to_thursday_8pm_ist():
    now = datetime(2026, 9, 3, 12, 0, tzinfo=timezone.utc)
    cutoff = scheduled_cutoff(now, "Asia/Kolkata")
    assert cutoff.weekday() == 3
    assert cutoff.hour == 20
    assert cutoff.minute == 0


def test_weekly_cutoff_uses_upcoming_thursday_before_run_day():
    now = datetime(2026, 9, 2, 7, 30, tzinfo=timezone.utc)
    cutoff = scheduled_cutoff(now, "Asia/Kolkata")
    assert cutoff.date().isoformat() == "2026-09-03"


def test_weekly_cutoff_uses_just_finished_thursday_after_run_day():
    now = datetime(2026, 9, 4, 7, 30, tzinfo=timezone.utc)
    cutoff = scheduled_cutoff(now, "Asia/Kolkata")
    assert cutoff.date().isoformat() == "2026-09-03"


def test_newsletter_template_uses_friday_after_thursday_cutoff():
    cutoff = scheduled_cutoff(
        datetime(2026, 9, 2, 7, 30, tzinfo=timezone.utc), "Asia/Kolkata"
    )
    context = run_context(
        workspace_id="default",
        profile_id="migraine",
        topic="migraine research",
        cutoff=cutoff,
    )
    assert _template_values(context, "migraine", 4)["issue_date"] == "September 4, 2026"
