from __future__ import annotations

import asyncio
import json
import logging
from dataclasses import dataclass, replace
from datetime import datetime, timedelta
from pathlib import Path
from typing import Sequence
from zoneinfo import ZoneInfo

from newsletter_agent.config import AppConfig, ProfileConfig
from newsletter_agent.domain.models import (
    ClassifiedItem,
    NewsletterIssue,
    RunContext,
)
from newsletter_agent.domain.scheduling import publication_date
from newsletter_agent.infrastructure.enrichment import deduplicate_enriched, filter_dead_links
from newsletter_agent.ports import (
    AlertSource,
    Categorizer,
    ContentEnricher,
    DraftPublisher,
    EmbeddingProvider,
    NewsletterRenderer,
    ResearchRepository,
)


logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class ProfileRunResult:
    profile_id: str
    state: str
    result_count: int = 0
    draft_id: str | None = None
    preview_html: str | None = None
    error: str | None = None


def scheduled_cutoff(now: datetime, timezone_name: str) -> datetime:
    timezone = ZoneInfo(timezone_name)
    local_now = now.astimezone(timezone)
    # Processing runs Thursday evening. The reader-facing issue date is the
    # following Friday and is calculated separately from this ingestion cutoff.
    days_to_thursday = 3 - local_now.weekday()
    candidate = (local_now + timedelta(days=days_to_thursday)).replace(
        hour=20, minute=0, second=0, microsecond=0
    )
    return candidate


def run_context(
    *,
    workspace_id: str,
    profile_id: str,
    topic: str,
    cutoff: datetime,
    lookback_days: int = 7,
) -> RunContext:
    issue_date = cutoff.date().isoformat()
    return RunContext(
        workspace_id=workspace_id,
        profile_id=profile_id,
        topic=topic,
        window_start=cutoff - timedelta(days=lookback_days),
        cutoff=cutoff,
        issue_key=f"{workspace_id}:{profile_id}:{issue_date}",
    )


def _template_values(
    context: RunContext, profile_id: str, result_count: int
) -> dict[str, str]:
    published = publication_date(context.cutoff)
    issue_date = f"{published.strftime('%B')} {published.day}, {published.year}"
    return {
        "topic": context.topic,
        "issue_date": issue_date,
        "result_count": str(result_count),
        "profile_id": profile_id,
    }


class NewsletterPipeline:
    def __init__(
        self,
        *,
        config: AppConfig,
        alert_source: AlertSource,
        enricher: ContentEnricher,
        categorizer: Categorizer,
        renderer: NewsletterRenderer,
        publisher: DraftPublisher,
        embedding_provider: EmbeddingProvider | None = None,
        repository: ResearchRepository | None = None,
    ) -> None:
        self.config = config
        self.alert_source = alert_source
        self.enricher = enricher
        self.categorizer = categorizer
        self.renderer = renderer
        self.publisher = publisher
        self.embedding_provider = embedding_provider
        self.repository = repository
        self.profile_semaphore = asyncio.Semaphore(config.runtime.max_profile_concurrency)

    async def _classify_all(
        self, enriched_items, profile: ProfileConfig
    ) -> tuple[ClassifiedItem, ...]:
        tasks = [self.categorizer.classify(item, profile) for item in enriched_items]
        return tuple(await asyncio.gather(*tasks))

    async def _persist_and_embed(
        self,
        context: RunContext,
        issue_id: str,
        items: Sequence[ClassifiedItem],
    ) -> None:
        if not self.repository or not self.embedding_provider:
            return
        work: list[tuple[str, str]] = []
        for position, item in enumerate(items):
            _, embedding_id, needs_embedding = await self.repository.persist_item(
                context,
                issue_id,
                item,
                position,
                self.embedding_provider.provider_name,
                self.embedding_provider.model_name,
            )
            if needs_embedding:
                work.append((embedding_id, item.enriched.source_text))
        if not work:
            return
        try:
            vectors = await self.embedding_provider.embed([text for _, text in work])
            for (embedding_id, _), vector in zip(work, vectors, strict=True):
                await self.repository.save_embedding(embedding_id, vector)
        except Exception as exc:
            logger.warning(
                "embedding_batch_failed profile=%s count=%s error=%s",
                context.profile_id,
                len(work),
                type(exc).__name__,
            )
            await asyncio.gather(
                *(
                    self.repository.mark_embedding_failed(
                        embedding_id, f"{type(exc).__name__}: {exc}"
                    )
                    for embedding_id, _ in work
                )
            )

    async def run_profile(
        self,
        profile_id: str,
        profile: ProfileConfig,
        *,
        cutoff: datetime,
        dry_run: bool,
        keep_labels: bool,
        lookback_days: int,
    ) -> ProfileRunResult:
        async with self.profile_semaphore:
            context = run_context(
                workspace_id=self.config.runtime.workspace_id,
                profile_id=profile_id,
                topic=profile.topic,
                cutoff=cutoff,
                lookback_days=lookback_days,
            )
            # A label-preserving inspection run must not claim the production
            # issue key. Otherwise an early manual run could cause Thursday's
            # scheduled production run to see the issue as already completed.
            if keep_labels:
                context = replace(
                    context,
                    issue_key=f"{context.issue_key}:labels-kept:{lookback_days}d",
                )
            persisted = None
            try:
                batch = await self.alert_source.list_pending(context, profile)
                if not dry_run:
                    if not self.repository:
                        raise RuntimeError("a repository is required for non-dry runs")
                    persisted = await self.repository.begin_or_resume(context, profile, batch)
                    if persisted.state == "completed":
                        return ProfileRunResult(
                            profile_id=profile_id,
                            state="already_completed",
                            draft_id=persisted.draft_id,
                        )
                    if persisted.draft_id:
                        if not keep_labels:
                            message_ids = await self.repository.pending_message_ids(
                                persisted.issue_id
                            )
                            await self.alert_source.acknowledge(message_ids, batch.label_id)
                            await self.repository.mark_messages_acknowledged(
                                persisted.issue_id, message_ids
                            )
                        await self.repository.complete_issue(persisted.issue_id)
                        return ProfileRunResult(
                            profile_id=profile_id,
                            state=("recovered_labels_kept" if keep_labels else "recovered"),
                            draft_id=persisted.draft_id,
                        )

                enriched = await asyncio.gather(
                    *(self.enricher.enrich(item) for item in batch.items)
                )
                live_enriched = filter_dead_links(list(enriched))
                if len(live_enriched) != len(enriched):
                    logger.info(
                        "dead_research_links_removed profile=%s before=%s after=%s removed=%s",
                        profile_id,
                        len(enriched),
                        len(live_enriched),
                        len(enriched) - len(live_enriched),
                    )
                unique_enriched = deduplicate_enriched(live_enriched)
                if len(unique_enriched) != len(live_enriched):
                    logger.info(
                        "research_results_deduplicated profile=%s before=%s after=%s removed=%s",
                        profile_id,
                        len(live_enriched),
                        len(unique_enriched),
                        len(live_enriched) - len(unique_enriched),
                    )
                classified = await self._classify_all(unique_enriched, profile)
                summary = await self.categorizer.summarize_issue(classified, profile)
                values = _template_values(context, profile_id, len(classified))
                issue = NewsletterIssue(
                    context=context,
                    subject=profile.newsletter.subject_template.format_map(values),
                    title=profile.newsletter.title_template.format_map(values),
                    summary=summary,
                    items=classified,
                    recipients=profile.newsletter.recipients,
                    accent_color=profile.newsletter.accent_color,
                    subtitle=profile.newsletter.subtitle,
                    footer=profile.newsletter.footer,
                )
                rendered = self.renderer.render(issue, profile)

                if dry_run:
                    directory = Path(self.config.runtime.preview_directory)
                    directory.mkdir(parents=True, exist_ok=True)
                    html_path = directory / f"{profile_id}-{context.cutoff.date().isoformat()}.html"
                    text_path = directory / f"{profile_id}-{context.cutoff.date().isoformat()}.txt"
                    html_path.write_text(rendered.html, encoding="utf-8")
                    text_path.write_text(rendered.text, encoding="utf-8")
                    return ProfileRunResult(
                        profile_id=profile_id,
                        state="previewed",
                        result_count=len(classified),
                        preview_html=str(html_path.resolve()),
                    )

                assert persisted is not None and self.repository is not None
                if profile.rag_indexing_enabled:
                    await self._persist_and_embed(context, persisted.issue_id, classified)
                else:
                    logger.info(
                        "rag_indexing_skipped profile=%s count=%s",
                        profile_id,
                        len(classified),
                    )
                draft_id = await self.publisher.create_or_find(
                    rendered,
                    context.issue_key,
                    persisted.draft_id,
                )
                await self.repository.record_draft(
                    persisted.issue_id, draft_id, rendered.subject, summary
                )
                if not keep_labels:
                    message_ids = await self.repository.pending_message_ids(persisted.issue_id)
                    await self.alert_source.acknowledge(message_ids, batch.label_id)
                    await self.repository.mark_messages_acknowledged(
                        persisted.issue_id, message_ids
                    )
                else:
                    logger.info(
                        "gmail_labels_kept profile=%s issue_key=%s",
                        profile_id,
                        context.issue_key,
                    )
                await self.repository.complete_issue(persisted.issue_id)
                return ProfileRunResult(
                    profile_id=profile_id,
                    state=("completed_labels_kept" if keep_labels else "completed"),
                    result_count=len(classified),
                    draft_id=draft_id,
                )
            except Exception as exc:
                if persisted and self.repository:
                    try:
                        await self.repository.fail_issue(
                            persisted.issue_id, f"{type(exc).__name__}: {exc}"
                        )
                    except Exception:
                        logger.exception("failed_to_record_profile_error profile=%s", profile_id)
                logger.exception("profile_run_failed profile=%s", profile_id)
                return ProfileRunResult(
                    profile_id=profile_id,
                    state="failed",
                    error=f"{type(exc).__name__}: {exc}",
                )

    async def run(
        self,
        *,
        profile_ids: Sequence[str] | None = None,
        dry_run: bool = False,
        keep_labels: bool = False,
        lookback_days: int | None = None,
        now: datetime | None = None,
    ) -> list[ProfileRunResult]:
        timezone = ZoneInfo(self.config.runtime.timezone)
        current = now or datetime.now(timezone)
        cutoff = scheduled_cutoff(current, self.config.runtime.timezone)
        effective_lookback = (
            lookback_days
            if lookback_days is not None
            else self.config.runtime.alert_lookback_days
        )
        if not 1 <= effective_lookback <= 90:
            raise ValueError("lookback_days must be between 1 and 90")
        selected = set(profile_ids or [])
        if selected:
            missing = selected - self.config.profiles.keys()
            if missing:
                raise ValueError(f"unknown profiles: {sorted(missing)}")
        profiles = [
            (profile_id, profile)
            for profile_id, profile in self.config.profiles.items()
            if profile.enabled and (not selected or profile_id in selected)
        ]
        if not dry_run and self.repository and self.embedding_provider:
            try:
                await self.retry_embeddings(
                    limit=self.config.runtime.embedding_batch_size * 4
                )
            except Exception as exc:
                # Corpus backfill is deliberately non-blocking for newsletter production.
                logger.warning(
                    "pending_embedding_retry_failed error=%s",
                    type(exc).__name__,
                )
        return list(
            await asyncio.gather(
                *(
                    self.run_profile(
                        profile_id,
                        profile,
                        cutoff=cutoff,
                        dry_run=dry_run,
                        keep_labels=keep_labels,
                        lookback_days=effective_lookback,
                    )
                    for profile_id, profile in profiles
                )
            )
        )

    async def retry_embeddings(self, limit: int = 200) -> int:
        if not self.repository or not self.embedding_provider:
            raise RuntimeError("repository and embedding provider are required")
        pending = await self.repository.list_pending_embeddings(limit)
        if not pending:
            return 0
        vectors = await self.embedding_provider.embed([row.text for row in pending])
        for row, vector in zip(pending, vectors, strict=True):
            await self.repository.save_embedding(row.embedding_id, vector)
        return len(pending)


def log_results(results: Sequence[ProfileRunResult]) -> None:
    for result in results:
        logger.info(
            "profile_result %s",
            json.dumps(
                {
                    "profile_id": result.profile_id,
                    "state": result.state,
                    "result_count": result.result_count,
                    "draft_id": result.draft_id,
                    "preview_html": result.preview_html,
                    "error": result.error,
                },
                sort_keys=True,
            ),
        )
