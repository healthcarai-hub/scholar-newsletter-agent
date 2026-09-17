from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any


FALLBACK_CATEGORY = "Additional Research"


@dataclass(frozen=True, slots=True)
class CategoryDefinition:
    name: str
    guidance: str


@dataclass(frozen=True, slots=True)
class RunContext:
    workspace_id: str
    profile_id: str
    topic: str
    window_start: datetime
    cutoff: datetime
    issue_key: str


@dataclass(frozen=True, slots=True)
class SourceMessage:
    message_id: str
    thread_id: str | None
    internal_date: datetime
    subject: str


@dataclass(frozen=True, slots=True)
class ResearchItem:
    title: str
    url: str
    snippet: str
    source_message_id: str
    authors: str | None = None
    venue: str | None = None
    published_at: str | None = None
    doi: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class AlertBatch:
    messages: tuple[SourceMessage, ...]
    items: tuple[ResearchItem, ...]
    label_id: str


@dataclass(frozen=True, slots=True)
class EnrichedItem:
    item: ResearchItem
    canonical_url: str
    fingerprint: str
    source_text: str
    abstract: str | None = None
    authors: str | None = None
    venue: str | None = None
    published_at: str | None = None
    doi: str | None = None
    enrichment_status: str = "alert_only"
    link_status: str = "unverified"
    http_status: int | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class ClassifiedItem:
    enriched: EnrichedItem
    category: str
    headline: str
    brief: str
    source_label: str | None = None
    source_type: str = "unknown"
    priority: str = "normal"


@dataclass(frozen=True, slots=True)
class NewsletterIssue:
    context: RunContext
    subject: str
    title: str
    summary: str
    items: tuple[ClassifiedItem, ...]
    recipients: tuple[str, ...]
    accent_color: str
    subtitle: str
    footer: str


@dataclass(frozen=True, slots=True)
class RenderedNewsletter:
    subject: str
    html: str
    text: str
    recipients: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class PersistedIssue:
    issue_id: str
    draft_id: str | None
    state: str
    lease_token: str


@dataclass(frozen=True, slots=True)
class PendingEmbedding:
    embedding_id: str
    chunk_id: str
    text: str
