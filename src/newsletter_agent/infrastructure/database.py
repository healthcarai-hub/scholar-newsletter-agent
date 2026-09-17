from __future__ import annotations

import asyncio
import hashlib
import json
import uuid
from datetime import datetime, timedelta, timezone
from typing import Any, Sequence

from sqlalchemy import (
    JSON,
    Boolean,
    Column,
    DateTime,
    ForeignKey,
    Integer,
    MetaData,
    String,
    Table,
    Text,
    UniqueConstraint,
    and_,
    create_engine,
    insert,
    select,
    update,
)
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.engine import Engine
from sqlalchemy.types import UserDefinedType

from newsletter_agent.config import ProfileConfig
from newsletter_agent.domain.models import (
    AlertBatch,
    ClassifiedItem,
    PendingEmbedding,
    PersistedIssue,
    RunContext,
)


class VectorType(UserDefinedType):
    cache_ok = True

    def get_col_spec(self, **kw: Any) -> str:
        return "vector"

    def bind_processor(self, dialect):
        def process(value):
            if value is None or isinstance(value, str):
                return value
            return "[" + ",".join(str(float(number)) for number in value) + "]"

        return process


metadata = MetaData()

workspaces = Table(
    "workspaces",
    metadata,
    Column("id", String(120), primary_key=True),
    Column("name", String(240), nullable=False),
    Column("created_at", DateTime(timezone=True), nullable=False),
)

newsletter_profiles = Table(
    "newsletter_profiles",
    metadata,
    Column("workspace_id", String(120), ForeignKey("workspaces.id"), primary_key=True),
    Column("profile_id", String(120), primary_key=True),
    Column("topic", String(240), nullable=False),
    Column("enabled", Boolean, nullable=False),
    Column("rag_indexing_enabled", Boolean, nullable=False),
    Column("config_hash", String(64), nullable=False),
    Column("config_snapshot", JSON, nullable=False),
    Column("updated_at", DateTime(timezone=True), nullable=False),
)

research_items = Table(
    "research_items",
    metadata,
    Column("id", String(36), primary_key=True),
    Column("fingerprint", String(64), nullable=False, unique=True),
    Column("canonical_url", Text, nullable=False),
    Column("doi", String(300)),
    Column("title", Text, nullable=False),
    Column("snippet", Text, nullable=False),
    Column("abstract", Text),
    Column("authors", Text),
    Column("venue", Text),
    Column("published_at", String(120)),
    Column("metadata", JSON, nullable=False),
    Column("created_at", DateTime(timezone=True), nullable=False),
    Column("updated_at", DateTime(timezone=True), nullable=False),
)

profile_research_items = Table(
    "profile_research_items",
    metadata,
    Column("workspace_id", String(120), nullable=False),
    Column("profile_id", String(120), nullable=False),
    Column("research_item_id", String(36), ForeignKey("research_items.id"), nullable=False),
    Column("first_seen_at", DateTime(timezone=True), nullable=False),
    UniqueConstraint("workspace_id", "profile_id", "research_item_id"),
)

document_chunks = Table(
    "document_chunks",
    metadata,
    Column("id", String(36), primary_key=True),
    Column("research_item_id", String(36), ForeignKey("research_items.id"), nullable=False),
    Column("chunk_index", Integer, nullable=False),
    Column("content", Text, nullable=False),
    Column("content_hash", String(64), nullable=False),
    Column("provenance", JSON, nullable=False),
    Column("created_at", DateTime(timezone=True), nullable=False),
    UniqueConstraint("research_item_id", "content_hash"),
)

embeddings = Table(
    "embeddings",
    metadata,
    Column("id", String(36), primary_key=True),
    Column("chunk_id", String(36), ForeignKey("document_chunks.id"), nullable=False),
    Column("provider", String(120), nullable=False),
    Column("model", String(240), nullable=False),
    Column("dimensions", Integer),
    Column("embedding", VectorType),
    Column("status", String(30), nullable=False),
    Column("error", Text),
    Column("created_at", DateTime(timezone=True), nullable=False),
    Column("updated_at", DateTime(timezone=True), nullable=False),
    UniqueConstraint("chunk_id", "provider", "model"),
)

newsletter_issues = Table(
    "newsletter_issues",
    metadata,
    Column("id", String(36), primary_key=True),
    Column("workspace_id", String(120), nullable=False),
    Column("profile_id", String(120), nullable=False),
    Column("issue_key", String(300), nullable=False),
    Column("cutoff", DateTime(timezone=True), nullable=False),
    Column("state", String(40), nullable=False),
    Column("subject", Text),
    Column("summary", Text),
    Column("draft_id", String(300)),
    Column("lease_token", String(36), nullable=False),
    Column("lease_expires_at", DateTime(timezone=True), nullable=False),
    Column("error", Text),
    Column("created_at", DateTime(timezone=True), nullable=False),
    Column("updated_at", DateTime(timezone=True), nullable=False),
    UniqueConstraint("workspace_id", "profile_id", "issue_key"),
)

issue_items = Table(
    "issue_items",
    metadata,
    Column("issue_id", String(36), ForeignKey("newsletter_issues.id"), nullable=False),
    Column("research_item_id", String(36), ForeignKey("research_items.id"), nullable=False),
    Column("category", String(240), nullable=False),
    Column("headline", Text, nullable=False),
    Column("brief", Text, nullable=False),
    Column("source_label", Text),
    Column("position", Integer, nullable=False),
    UniqueConstraint("issue_id", "research_item_id"),
)

source_messages = Table(
    "source_messages",
    metadata,
    Column("issue_id", String(36), ForeignKey("newsletter_issues.id"), nullable=False),
    Column("message_id", String(300), nullable=False),
    Column("thread_id", String(300)),
    Column("subject", Text, nullable=False),
    Column("internal_date", DateTime(timezone=True), nullable=False),
    Column("label_id", String(300), nullable=False),
    Column("acknowledged_at", DateTime(timezone=True)),
    UniqueConstraint("issue_id", "message_id"),
)


def normalize_database_url(url: str) -> str:
    if url.startswith("postgres://"):
        return "postgresql+psycopg://" + url[len("postgres://") :]
    if url.startswith("postgresql://") and "+" not in url.split(":", 1)[0]:
        return "postgresql+psycopg://" + url[len("postgresql://") :]
    return url


class PostgresRepository:
    def __init__(self, database_url: str, *, lease_minutes: int = 120) -> None:
        self.engine: Engine = create_engine(normalize_database_url(database_url), pool_pre_ping=True)
        self.lease_minutes = lease_minutes

    @staticmethod
    def _now() -> datetime:
        return datetime.now(timezone.utc)

    def _begin_or_resume_sync(
        self, context: RunContext, profile: ProfileConfig, batch: AlertBatch
    ) -> PersistedIssue:
        now = self._now()
        lease_token = str(uuid.uuid4())
        config_snapshot = profile.model_dump(mode="json")
        config_hash = hashlib.sha256(
            json.dumps(config_snapshot, sort_keys=True).encode("utf-8")
        ).hexdigest()
        with self.engine.begin() as connection:
            connection.execute(
                pg_insert(workspaces)
                .values(id=context.workspace_id, name=context.workspace_id, created_at=now)
                .on_conflict_do_nothing(index_elements=[workspaces.c.id])
            )
            connection.execute(
                pg_insert(newsletter_profiles)
                .values(
                    workspace_id=context.workspace_id,
                    profile_id=context.profile_id,
                    topic=profile.topic,
                    enabled=profile.enabled,
                    rag_indexing_enabled=profile.rag_indexing_enabled,
                    config_hash=config_hash,
                    config_snapshot=config_snapshot,
                    updated_at=now,
                )
                .on_conflict_do_update(
                    index_elements=[newsletter_profiles.c.workspace_id, newsletter_profiles.c.profile_id],
                    set_={
                        "topic": profile.topic,
                        "enabled": profile.enabled,
                        "rag_indexing_enabled": profile.rag_indexing_enabled,
                        "config_hash": config_hash,
                        "config_snapshot": config_snapshot,
                        "updated_at": now,
                    },
                )
            )
            existing = connection.execute(
                select(newsletter_issues).where(
                    and_(
                        newsletter_issues.c.workspace_id == context.workspace_id,
                        newsletter_issues.c.profile_id == context.profile_id,
                        newsletter_issues.c.issue_key == context.issue_key,
                    )
                )
            ).mappings().first()
            if existing:
                if existing["state"] == "completed":
                    return PersistedIssue(
                        issue_id=existing["id"],
                        draft_id=existing["draft_id"],
                        state="completed",
                        lease_token=existing["lease_token"],
                    )
                if (
                    existing["state"] in {"processing", "draft_created"}
                    and existing["lease_expires_at"] > now
                ):
                    raise RuntimeError(
                        f"profile {context.profile_id} already has an active run for {context.issue_key}"
                    )
                claimed = connection.execute(
                    update(newsletter_issues)
                    .where(
                        and_(
                            newsletter_issues.c.id == existing["id"],
                            newsletter_issues.c.state == existing["state"],
                            newsletter_issues.c.lease_token == existing["lease_token"],
                        )
                    )
                    .values(
                        state="processing",
                        lease_token=lease_token,
                        lease_expires_at=now + timedelta(minutes=self.lease_minutes),
                        error=None,
                        updated_at=now,
                    )
                )
                if claimed.rowcount != 1:
                    raise RuntimeError(
                        f"profile {context.profile_id} run was claimed concurrently"
                    )
                issue_id = existing["id"]
                draft_id = existing["draft_id"]
            else:
                issue_id = str(uuid.uuid4())
                draft_id = None
                connection.execute(
                    insert(newsletter_issues).values(
                        id=issue_id,
                        workspace_id=context.workspace_id,
                        profile_id=context.profile_id,
                        issue_key=context.issue_key,
                        cutoff=context.cutoff,
                        state="processing",
                        lease_token=lease_token,
                        lease_expires_at=now + timedelta(minutes=self.lease_minutes),
                        created_at=now,
                        updated_at=now,
                    )
                )
            if not draft_id:
                for message in batch.messages:
                    connection.execute(
                        pg_insert(source_messages)
                        .values(
                            issue_id=issue_id,
                            message_id=message.message_id,
                            thread_id=message.thread_id,
                            subject=message.subject,
                            internal_date=message.internal_date,
                            label_id=batch.label_id,
                        )
                        .on_conflict_do_nothing(
                            index_elements=[source_messages.c.issue_id, source_messages.c.message_id]
                        )
                    )
            return PersistedIssue(
                issue_id=issue_id,
                draft_id=draft_id,
                state="processing",
                lease_token=lease_token,
            )

    async def begin_or_resume(
        self, context: RunContext, profile: ProfileConfig, batch: AlertBatch
    ) -> PersistedIssue:
        return await asyncio.to_thread(self._begin_or_resume_sync, context, profile, batch)

    def _persist_item_sync(
        self,
        context: RunContext,
        issue_id: str,
        classified: ClassifiedItem,
        position: int,
        embedding_provider: str,
        embedding_model: str,
    ) -> tuple[str, str, bool]:
        now = self._now()
        item = classified.enriched
        with self.engine.begin() as connection:
            item_id = connection.execute(
                select(research_items.c.id).where(
                    research_items.c.fingerprint == item.fingerprint
                )
            ).scalar_one_or_none()
            if not item_id:
                item_id = str(uuid.uuid4())
                connection.execute(
                    insert(research_items).values(
                        id=item_id,
                        fingerprint=item.fingerprint,
                        canonical_url=item.canonical_url,
                        doi=item.doi,
                        title=item.item.title,
                        snippet=item.item.snippet,
                        abstract=item.abstract,
                        authors=item.authors,
                        venue=item.venue,
                        published_at=item.published_at,
                        metadata=item.metadata,
                        created_at=now,
                        updated_at=now,
                    )
                )
            else:
                connection.execute(
                    update(research_items)
                    .where(research_items.c.id == item_id)
                    .values(
                        canonical_url=item.canonical_url,
                        doi=item.doi,
                        title=item.item.title,
                        snippet=item.item.snippet,
                        abstract=item.abstract,
                        authors=item.authors,
                        venue=item.venue,
                        published_at=item.published_at,
                        metadata=item.metadata,
                        updated_at=now,
                    )
                )
            connection.execute(
                pg_insert(profile_research_items)
                .values(
                    workspace_id=context.workspace_id,
                    profile_id=context.profile_id,
                    research_item_id=item_id,
                    first_seen_at=now,
                )
                .on_conflict_do_nothing(
                    index_elements=[
                        profile_research_items.c.workspace_id,
                        profile_research_items.c.profile_id,
                        profile_research_items.c.research_item_id,
                    ]
                )
            )
            content_hash = hashlib.sha256(item.source_text.encode("utf-8")).hexdigest()
            chunk_id = connection.execute(
                select(document_chunks.c.id).where(
                    and_(
                        document_chunks.c.research_item_id == item_id,
                        document_chunks.c.content_hash == content_hash,
                    )
                )
            ).scalar_one_or_none()
            if not chunk_id:
                chunk_id = str(uuid.uuid4())
                connection.execute(
                    insert(document_chunks).values(
                        id=chunk_id,
                        research_item_id=item_id,
                        chunk_index=0,
                        content=item.source_text,
                        content_hash=content_hash,
                        provenance={
                            "profile_id": context.profile_id,
                            "source_message_id": item.item.source_message_id,
                            "enrichment_status": item.enrichment_status,
                        },
                        created_at=now,
                    )
                )
            embedding_row = connection.execute(
                select(embeddings.c.id, embeddings.c.status).where(
                    and_(
                        embeddings.c.chunk_id == chunk_id,
                        embeddings.c.provider == embedding_provider,
                        embeddings.c.model == embedding_model,
                    )
                )
            ).mappings().first()
            if not embedding_row:
                embedding_id = str(uuid.uuid4())
                needs_embedding = True
                connection.execute(
                    insert(embeddings).values(
                        id=embedding_id,
                        chunk_id=chunk_id,
                        provider=embedding_provider,
                        model=embedding_model,
                        status="pending",
                        created_at=now,
                        updated_at=now,
                    )
                )
            else:
                embedding_id = embedding_row["id"]
                needs_embedding = embedding_row["status"] != "ready"
            connection.execute(
                pg_insert(issue_items)
                .values(
                    issue_id=issue_id,
                    research_item_id=item_id,
                    category=classified.category,
                    headline=classified.headline,
                    brief=classified.brief,
                    source_label=classified.source_label,
                    position=position,
                )
                .on_conflict_do_update(
                    index_elements=[issue_items.c.issue_id, issue_items.c.research_item_id],
                    set_={
                        "category": classified.category,
                        "headline": classified.headline,
                        "brief": classified.brief,
                        "source_label": classified.source_label,
                        "position": position,
                    },
                )
            )
        return chunk_id, embedding_id, needs_embedding

    async def persist_item(
        self,
        context: RunContext,
        issue_id: str,
        item: ClassifiedItem,
        position: int,
        embedding_provider: str,
        embedding_model: str,
    ) -> tuple[str, str, bool]:
        return await asyncio.to_thread(
            self._persist_item_sync,
            context,
            issue_id,
            item,
            position,
            embedding_provider,
            embedding_model,
        )

    def _save_embedding_sync(self, embedding_id: str, vector: Sequence[float]) -> None:
        with self.engine.begin() as connection:
            connection.execute(
                update(embeddings)
                .where(embeddings.c.id == embedding_id)
                .values(
                    embedding=list(vector),
                    dimensions=len(vector),
                    status="ready",
                    error=None,
                    updated_at=self._now(),
                )
            )

    async def save_embedding(self, embedding_id: str, vector: Sequence[float]) -> None:
        await asyncio.to_thread(self._save_embedding_sync, embedding_id, vector)

    def _mark_embedding_failed_sync(self, embedding_id: str, error: str) -> None:
        with self.engine.begin() as connection:
            connection.execute(
                update(embeddings)
                .where(embeddings.c.id == embedding_id)
                .values(status="failed", error=error[:2000], updated_at=self._now())
            )

    async def mark_embedding_failed(self, embedding_id: str, error: str) -> None:
        await asyncio.to_thread(self._mark_embedding_failed_sync, embedding_id, error)

    def _record_draft_sync(
        self, issue_id: str, draft_id: str, subject: str, summary: str
    ) -> None:
        with self.engine.begin() as connection:
            connection.execute(
                update(newsletter_issues)
                .where(newsletter_issues.c.id == issue_id)
                .values(
                    state="draft_created",
                    draft_id=draft_id,
                    subject=subject,
                    summary=summary,
                    updated_at=self._now(),
                )
            )

    async def record_draft(
        self, issue_id: str, draft_id: str, subject: str, summary: str
    ) -> None:
        await asyncio.to_thread(
            self._record_draft_sync, issue_id, draft_id, subject, summary
        )

    def _set_issue_state_sync(self, issue_id: str, state: str, error: str | None = None) -> None:
        with self.engine.begin() as connection:
            connection.execute(
                update(newsletter_issues)
                .where(newsletter_issues.c.id == issue_id)
                .values(state=state, error=error, updated_at=self._now())
            )

    async def complete_issue(self, issue_id: str) -> None:
        await asyncio.to_thread(self._set_issue_state_sync, issue_id, "completed", None)

    async def fail_issue(self, issue_id: str, error: str) -> None:
        await asyncio.to_thread(self._set_issue_state_sync, issue_id, "failed", error[:4000])

    def _list_pending_embeddings_sync(self, limit: int) -> list[PendingEmbedding]:
        with self.engine.connect() as connection:
            rows = connection.execute(
                select(
                    embeddings.c.id.label("embedding_id"),
                    document_chunks.c.id.label("chunk_id"),
                    document_chunks.c.content,
                )
                .join(document_chunks, embeddings.c.chunk_id == document_chunks.c.id)
                .where(embeddings.c.status.in_(["pending", "failed"]))
                .order_by(embeddings.c.updated_at)
                .limit(limit)
            ).mappings()
            return [
                PendingEmbedding(
                    embedding_id=row["embedding_id"],
                    chunk_id=row["chunk_id"],
                    text=row["content"],
                )
                for row in rows
            ]

    async def list_pending_embeddings(self, limit: int) -> list[PendingEmbedding]:
        return await asyncio.to_thread(self._list_pending_embeddings_sync, limit)

    def close(self) -> None:
        self.engine.dispose()
