from __future__ import annotations

from typing import Protocol, Sequence

from newsletter_agent.config import ProfileConfig
from newsletter_agent.domain.models import (
    AlertBatch,
    ClassifiedItem,
    EnrichedItem,
    NewsletterIssue,
    PendingEmbedding,
    PersistedIssue,
    RenderedNewsletter,
    ResearchItem,
    RunContext,
)


class AlertSource(Protocol):
    async def list_pending(self, context: RunContext, profile: ProfileConfig) -> AlertBatch: ...

    async def acknowledge(self, message_ids: Sequence[str], label_id: str) -> None: ...


class ContentEnricher(Protocol):
    async def enrich(self, item: ResearchItem) -> EnrichedItem: ...


class Categorizer(Protocol):
    async def classify(self, item: EnrichedItem, profile: ProfileConfig) -> ClassifiedItem: ...

    async def summarize_issue(
        self, items: Sequence[ClassifiedItem], profile: ProfileConfig
    ) -> str: ...


class EmbeddingProvider(Protocol):
    provider_name: str
    model_name: str

    async def embed(self, texts: Sequence[str]) -> list[list[float]]: ...


class ResearchRepository(Protocol):
    async def begin_or_resume(
        self, context: RunContext, profile: ProfileConfig, batch: AlertBatch
    ) -> PersistedIssue: ...

    async def persist_item(
        self,
        context: RunContext,
        issue_id: str,
        item: ClassifiedItem,
        position: int,
        embedding_provider: str,
        embedding_model: str,
    ) -> tuple[str, str, bool]: ...

    async def save_embedding(
        self, embedding_id: str, vector: Sequence[float]
    ) -> None: ...

    async def mark_embedding_failed(self, embedding_id: str, error: str) -> None: ...

    async def record_draft(
        self, issue_id: str, draft_id: str, subject: str, summary: str
    ) -> None: ...

    async def pending_message_ids(self, issue_id: str) -> list[str]: ...

    async def mark_messages_acknowledged(
        self, issue_id: str, message_ids: Sequence[str]
    ) -> None: ...

    async def complete_issue(self, issue_id: str) -> None: ...

    async def fail_issue(self, issue_id: str, error: str) -> None: ...

    async def list_pending_embeddings(self, limit: int) -> list[PendingEmbedding]: ...


class NewsletterRenderer(Protocol):
    def render(self, issue: NewsletterIssue, profile: ProfileConfig) -> RenderedNewsletter: ...


class DraftPublisher(Protocol):
    async def create_or_find(
        self,
        rendered: RenderedNewsletter,
        issue_key: str,
        existing_draft_id: str | None = None,
    ) -> str: ...
