# Scholar Newsletter Agent

A multi-topic Python agent that reads labeled Google Scholar alert emails, enriches and
categorizes their results, stores source-grounded vectors in PostgreSQL/pgvector, and creates
reviewable Gmail drafts. It never sends email.

## What version 1 does

- Processes every enabled topic profile independently.
- Uses `{topic}` and structured `{categories}` values in editable prompt templates.
- Reads one Gmail label and creates one Gmail draft per profile.
- Reads only alerts inside the configured rolling window (`alert_lookback_days`, seven by default).
- Omits every category that has no results, including from the navigation.
- Orders populated categories by descending result count, with configured order as the tie-breaker.
- Runs on Thursday evening but uses the following Friday as the displayed issue date.
- Supports per-profile greeting, introduction, closing, and signature copy.
- Places otherwise unmatched items in the non-empty-only `Additional Research` fallback section.
- Extracts public citation metadata with safe limits and falls back to Scholar snippets.
- Removes confirmed HTTP 404/410 and soft-404 links before categorization while retaining
  bot-blocked or temporarily unreachable links as unverified.
- Deduplicates before categorization using DOI, canonical URL, normalized title, and
  conservative abstract/snippet similarity across differently titled copies.
- Keeps bibliographic metadata internally but omits the DOI/journal/date metadata line from cards.
- Stores canonical papers, profile associations, source chunks, and embeddings for future RAG.
- Recovers an interrupted run without producing a second draft.
- Isolates profile failures so another topic can still complete.

## Local setup

Python 3.12 is required.

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install '.[dev]'
cp .env.example .env
newsletter-agent validate-config
newsletter-agent sample-preview --output outputs
```

Edit `config.yaml` to set topics, Gmail label names, categories, prompts, recipients, greeting,
introduction, closing, signature, and branding. The configuration contains no credentials and is
safe to commit. Keep all secrets in environment variables.

Use the research subject itself for `topic` (for example, `Migraine` or `Cardiology`). The supplied
newsletter convention adds “Research” through `Weekly {topic} Research Digest` and
`This Week in {topic} Research`, avoiding duplicated wording such as “Research Research.”

After setting `DATABASE_URL` in `.env`, initialize or upgrade the PostgreSQL/pgvector schema with
`alembic upgrade head`. Local Alembic runs load the project `.env` automatically; Railway-provided
environment variables retain precedence during deployment.

### Prompt placeholders

The validator permits only:

- Prompt templates: `{topic}`, `{categories}`, `{source_text}`, `{items}`, `{issue_date}`,
  `{result_count}`.
- Subject/title templates: `{topic}`, `{issue_date}`, `{result_count}`, `{profile_id}`.

Required placeholders vary by prompt. `validate-config` reports an error before any Gmail or
provider call if a required field is missing or an unknown field is present.

## Gmail OAuth

1. Create a Google Cloud project and enable the Gmail API.
2. Configure an OAuth consent screen and create a Desktop OAuth client.
3. Set `GOOGLE_CLIENT_ID` and `GOOGLE_CLIENT_SECRET` locally.
4. Run:

   ```bash
   newsletter-agent auth
   ```

5. Store the returned value as `GOOGLE_REFRESH_TOKEN` locally and in Railway.

Only `gmail.label_name` is accepted in each profile. At runtime the authenticated Gmail API
lists the account's labels, matches that name case-insensitively, and uses the returned Gmail
label ID for message queries and post-draft acknowledgement. Label IDs are never configured.

Local CLI commands load `.env` automatically. Quoting values is optional unless they contain
spaces, `#`, or leading/trailing whitespace; existing process and Railway environment variables
always take precedence over `.env`.

The application requests `gmail.modify`, which is needed to read alert bodies, create drafts,
and remove the configured label. There is intentionally no call to `users.messages.send` or
`users.drafts.send` anywhere in the codebase.

For a personal OAuth application, ensure the consent configuration is suitable for durable
offline access before relying on a weekly cloud job.

## Commands

```bash
# Validate config only; does not access external services.
newsletter-agent validate-config

# Render deterministic example newsletters; does not access external services.
newsletter-agent sample-preview --output outputs

# Authenticate, resolve configured labels, and parse Scholar alerts using Gmail GET calls only.
# This does not call the LLM, write drafts, remove labels, or access PostgreSQL.
newsletter-agent check-gmail
newsletter-agent check-gmail --profile migraine

# Read Gmail and call the chat provider, but do not write Gmail or PostgreSQL.
newsletter-agent run --dry-run

# Process one enabled profile.
newsletter-agent run --profile migraine

# Process normally and create the draft, but preserve all source Gmail labels.
newsletter-agent run --profile migraine --keep-labels

# Temporarily inspect a 10-day window; config.yaml remains at its seven-day default.
newsletter-agent run --keep-labels --lookback-days 10

# Process all enabled profiles.
newsletter-agent run

# Retry pending/failed corpus embeddings.
newsletter-agent retry-embeddings --limit 200
```

Dry-run previews are written under `runtime.preview_directory`. The weekly message window is
start-exclusive and cutoff-inclusive, preventing boundary messages from appearing twice. Old
labeled messages outside the window are ignored and left untouched. Processing uses the Thursday
8:00 PM IST cutoff, while the subject and newsletter display the following Friday. For example, a
Thursday, September 3 run produces a September 4 issue. A normal run follows this order:

1. Read and parse all labeled messages at or before the fixed weekly cutoff.
2. Claim or resume the issue in PostgreSQL.
3. Enrich, categorize, summarize, persist, and attempt embeddings.
4. Create or find the Gmail draft.
5. Persist the draft ID.
6. Remove the source label from the recorded message IDs.
7. Mark the issue complete.

Parsing, model, rendering, or draft failures leave Gmail labels untouched. Embedding failures are
stored for retry and do not block the newsletter. Every normal run first attempts a bounded
backfill of pending vectors; `retry-embeddings` is also available for an explicit larger backfill.
For an inspection run that should still persist data and create a real Gmail draft, add
`--keep-labels`; the resulting issue is completed without acknowledging or modifying its source
message labels. It uses a separate idempotent issue-key variant, so an inspection run does not
claim or block the corresponding scheduled production issue. `--lookback-days` can temporarily
override the configured window only when combined with `--keep-labels` or `--dry-run`; its value is
included in the inspection issue key.

For rate-limited OpenAI-compatible providers, use `max_llm_concurrency`,
`llm_request_interval_seconds`, `llm_max_retries`, and `llm_max_retry_wait_seconds`. The supplied
`config.yaml` uses one request at a time with a 2.1-second start interval for Groq's free tier.
HTTP 429 responses honor the provider's `retry-after` header. Categorization and newsletter copy
are generated together in one request per result to reduce request and token usage.

`embedding_max_input_chars` bounds each source passed to an embedding model while retaining the
full normalized source chunk in PostgreSQL. Fixed-dimension models can leave
`EMBEDDING_DIMENSIONS` blank; the returned vector length is recorded automatically.

## Railway deployment

1. Provision PostgreSQL with the `vector` extension available.
2. Deploy this repository as a service.
3. Add all values from `.env.example` in Railway Variables.
4. Review `config.yaml` and set real recipients only when wanted.
5. Deploy. The pre-deploy command applies Alembic migrations.

`railway.json` schedules `30 14 * * 4`, which is Thursday 14:30 UTC / 20:00 IST. The process is
one-shot and exits after all enabled profiles reach a terminal result.

## Adding another topic

Copy a profile in `config.yaml`, give it a unique profile ID and Gmail label, then change its
`topic`, categories, prompt wording, subject/title, recipients, and accent color. Validate before
deploying. Each enabled profile creates its own issue and draft.

## Future RAG and multi-user work

The current schema separates global research documents from workspace/profile associations.
`document_chunks` stores source text and provenance; `embeddings` supports multiple provider/model
versions. Future retrieval can filter through profile associations without changing ingestion or
rendering. A multi-user service can replace the config adapter with database-backed profile and
OAuth adapters while retaining the application ports and run context.
