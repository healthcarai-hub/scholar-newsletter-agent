from __future__ import annotations

import argparse
import asyncio
import logging
import os
import sys
from collections.abc import Sequence
from datetime import datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

from dotenv import load_dotenv

from newsletter_agent.application.pipeline import (
    NewsletterPipeline,
    log_results,
    run_context,
)
from newsletter_agent.config import AppConfig, EnvironmentSettings, load_config
from newsletter_agent.examples import render_sample_previews
from newsletter_agent.infrastructure.database import PostgresRepository
from newsletter_agent.infrastructure.enrichment import HttpMetadataEnricher
from newsletter_agent.infrastructure.gmail import (
    GmailAdapter,
    authorize_interactively,
    build_gmail_service,
    settings_for_auth,
)
from newsletter_agent.infrastructure.llm import (
    OpenAICompatibleCategorizer,
    OpenAICompatibleEmbeddingProvider,
)
from newsletter_agent.infrastructure.rendering import JinjaNewsletterRenderer


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="newsletter-agent")
    parser.add_argument(
        "--config",
        default=os.getenv("NEWSLETTER_CONFIG", "config.yaml"),
        help="Path to YAML configuration (default: config.yaml)",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("validate-config", help="Validate profiles and prompt templates")
    subparsers.add_parser("auth", help="Run local Gmail OAuth authorization")

    check_gmail = subparsers.add_parser(
        "check-gmail",
        help="Read and parse labeled Scholar alerts without making any changes",
    )
    check_gmail.add_argument("--profile", action="append", dest="profiles")

    run = subparsers.add_parser("run", help="Process enabled profiles")
    run.add_argument("--profile", action="append", dest="profiles")
    run.add_argument("--dry-run", action="store_true")
    run.add_argument(
        "--keep-labels",
        action="store_true",
        help=(
            "Run with an inspection issue key (legacy name; Gmail labels are always preserved)"
        ),
    )
    run.add_argument(
        "--lookback-days",
        type=int,
        help="Override the configured lookback for this run only (1-90)",
    )

    retry = subparsers.add_parser("retry-embeddings", help="Backfill pending vectors")
    retry.add_argument("--limit", type=int, default=200)

    sample = subparsers.add_parser("sample-preview", help="Render previews without external services")
    sample.add_argument("--output", default="outputs")
    return parser


def _require(value: str | None, name: str) -> str:
    if not value:
        raise RuntimeError(f"{name} is required")
    return value


def _rag_indexing_requested(
    config: AppConfig, profile_ids: Sequence[str] | None = None
) -> bool:
    selected = set(profile_ids or ())
    return any(
        profile.enabled
        and profile.rag_indexing_enabled
        and (not selected or profile_id in selected)
        for profile_id, profile in config.profiles.items()
    )


def _services(
    config: AppConfig,
    settings: EnvironmentSettings,
    *,
    dry_run: bool,
    profile_ids: Sequence[str] | None = None,
):
    gmail = GmailAdapter(build_gmail_service(settings))
    categorizer = OpenAICompatibleCategorizer(
        base_url=_require(settings.llm_base_url, "LLM_BASE_URL"),
        api_key=_require(settings.llm_api_key, "LLM_API_KEY"),
        model=_require(settings.llm_model, "LLM_MODEL"),
        max_concurrency=config.runtime.max_llm_concurrency,
        request_interval_seconds=config.runtime.llm_request_interval_seconds,
        retries=config.runtime.llm_max_retries,
        max_retry_wait_seconds=config.runtime.llm_max_retry_wait_seconds,
    )
    repository = None
    embedder = None
    if not dry_run:
        repository = PostgresRepository(_require(settings.database_url, "DATABASE_URL"))
    if not dry_run and _rag_indexing_requested(config, profile_ids):
        embedder = OpenAICompatibleEmbeddingProvider(
            base_url=_require(settings.embedding_base_url, "EMBEDDING_BASE_URL"),
            api_key=_require(settings.embedding_api_key, "EMBEDDING_API_KEY"),
            model=_require(settings.embedding_model, "EMBEDDING_MODEL"),
            dimensions=settings.embedding_dimensions,
            batch_size=config.runtime.embedding_batch_size,
            max_input_chars=config.runtime.embedding_max_input_chars,
        )
    pipeline = NewsletterPipeline(
        config=config,
        alert_source=gmail,
        enricher=HttpMetadataEnricher(
            timeout_seconds=config.runtime.request_timeout_seconds,
            max_page_bytes=config.runtime.max_page_bytes,
            max_concurrency=config.runtime.max_enrichment_concurrency,
        ),
        categorizer=categorizer,
        renderer=JinjaNewsletterRenderer(),
        publisher=gmail,
        embedding_provider=embedder,
        repository=repository,
    )
    return pipeline, repository


async def _run_command(args, config) -> int:
    if args.lookback_days is not None and not (args.keep_labels or args.dry_run):
        raise ValueError(
            "--lookback-days requires --keep-labels or --dry-run so the production window "
            "cannot be changed accidentally"
        )
    pipeline, repository = _services(
        config,
        EnvironmentSettings.from_env(),
        dry_run=args.dry_run,
        profile_ids=args.profiles,
    )
    try:
        results = await pipeline.run(
            profile_ids=args.profiles,
            dry_run=args.dry_run,
            keep_labels=args.keep_labels,
            lookback_days=args.lookback_days,
        )
        log_results(results)
        return 1 if any(result.state == "failed" for result in results) else 0
    finally:
        if repository:
            repository.close()


async def _check_gmail_command(args, config) -> int:
    """Verify OAuth, label resolution, and Scholar parsing using Gmail GET calls only."""
    gmail = GmailAdapter(build_gmail_service(EnvironmentSettings.from_env()))
    selected = set(args.profiles or ())
    unknown = selected - set(config.profiles)
    if unknown:
        raise ValueError(f"unknown profile(s): {', '.join(sorted(unknown))}")

    profiles = [
        (profile_id, profile)
        for profile_id, profile in config.profiles.items()
        if profile.enabled and (not selected or profile_id in selected)
    ]
    if not profiles:
        raise ValueError("no enabled profiles selected")

    # Diagnostics examine the latest rolling window. Production runs retain the
    # fixed Thursday cutoff so issue keys and recovery remain deterministic.
    cutoff = datetime.now(ZoneInfo(config.runtime.timezone))
    window_start = cutoff - timedelta(days=config.runtime.alert_lookback_days)
    failures = 0
    print(
        "Read-only Gmail check; window: "
        f"{window_start.isoformat()} < received <= {cutoff.isoformat()}"
    )
    for profile_id, profile in profiles:
        context = run_context(
            workspace_id=config.runtime.workspace_id,
            profile_id=profile_id,
            topic=profile.topic,
            cutoff=cutoff,
            lookback_days=config.runtime.alert_lookback_days,
        )
        try:
            batch = await gmail.list_pending(context, profile)
        except Exception as exc:
            failures += 1
            print(f"{profile_id}: FAILED — {exc}")
            continue
        print(
            f"{profile_id}: label resolved; "
            f"messages={len(batch.messages)}; results={len(batch.items)}"
        )
    print("Gmail mutations performed: 0")
    return 1 if failures else 0


async def _retry_command(args, config) -> int:
    settings = EnvironmentSettings.from_env()
    repository = PostgresRepository(_require(settings.database_url, "DATABASE_URL"))
    embedder = OpenAICompatibleEmbeddingProvider(
        base_url=_require(settings.embedding_base_url, "EMBEDDING_BASE_URL"),
        api_key=_require(settings.embedding_api_key, "EMBEDDING_API_KEY"),
        model=_require(settings.embedding_model, "EMBEDDING_MODEL"),
        dimensions=settings.embedding_dimensions,
        batch_size=config.runtime.embedding_batch_size,
        max_input_chars=config.runtime.embedding_max_input_chars,
    )
    try:
        pending = await repository.list_pending_embeddings(args.limit)
        if not pending:
            count = 0
        else:
            vectors = await embedder.embed([row.text for row in pending])
            for row, vector in zip(pending, vectors, strict=True):
                await repository.save_embedding(row.embedding_id, vector)
            count = len(pending)
        print(f"Embedded {count} pending document chunk(s).")
        return 0
    finally:
        if repository:
            repository.close()


def main(argv: list[str] | None = None) -> None:
    # Local runs load .env automatically. Railway-provided environment variables keep
    # precedence because python-dotenv does not override existing values by default.
    load_dotenv()
    args = _parser().parse_args(argv)
    logging.basicConfig(
        level=os.getenv("LOG_LEVEL", "INFO").upper(),
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    try:
        if args.command == "auth":
            client_id, client_secret = settings_for_auth()
            token = authorize_interactively(client_id, client_secret)
            print("Store this value as GOOGLE_REFRESH_TOKEN in Railway:")
            print(token)
            return

        config = load_config(args.config)
        if args.command == "validate-config":
            enabled = [name for name, profile in config.profiles.items() if profile.enabled]
            print(f"Configuration is valid. Enabled profiles: {', '.join(enabled) or 'none'}")
            return
        if args.command == "sample-preview":
            paths = render_sample_previews(config, args.output)
            for path in paths:
                print(path.resolve())
            return
        if args.command == "check-gmail":
            raise SystemExit(asyncio.run(_check_gmail_command(args, config)))
        if args.command == "run":
            raise SystemExit(asyncio.run(_run_command(args, config)))
        if args.command == "retry-embeddings":
            raise SystemExit(asyncio.run(_retry_command(args, config)))
    except (RuntimeError, ValueError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        raise SystemExit(2) from exc
