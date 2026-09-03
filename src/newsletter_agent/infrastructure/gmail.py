from __future__ import annotations

import asyncio
import base64
import os
from datetime import timezone
from email.message import EmailMessage
from email.utils import formatdate
from typing import Any, Sequence

from newsletter_agent.config import EnvironmentSettings, ProfileConfig
from newsletter_agent.domain.models import AlertBatch, RenderedNewsletter, RunContext, SourceMessage
from newsletter_agent.infrastructure.scholar_parser import (
    extract_mime_bodies,
    gmail_internal_date,
    parse_scholar_alert,
)


GMAIL_MODIFY_SCOPE = "https://www.googleapis.com/auth/gmail.modify"


def message_is_in_run_window(context: RunContext, internal_date) -> bool:
    received_utc = internal_date.astimezone(timezone.utc)
    return (
        context.window_start.astimezone(timezone.utc)
        < received_utc
        <= context.cutoff.astimezone(timezone.utc)
    )


def _headers(payload: dict[str, Any]) -> dict[str, str]:
    return {
        str(header.get("name", "")).casefold(): str(header.get("value", ""))
        for header in payload.get("headers", [])
    }


def build_gmail_service(settings: EnvironmentSettings):
    if not all(
        [
            settings.google_client_id,
            settings.google_client_secret,
            settings.google_refresh_token,
        ]
    ):
        raise RuntimeError(
            "GOOGLE_CLIENT_ID, GOOGLE_CLIENT_SECRET, and GOOGLE_REFRESH_TOKEN are required"
        )
    from google.oauth2.credentials import Credentials
    from googleapiclient.discovery import build

    credentials = Credentials(
        token=None,
        refresh_token=settings.google_refresh_token,
        token_uri="https://oauth2.googleapis.com/token",
        client_id=settings.google_client_id,
        client_secret=settings.google_client_secret,
        scopes=[GMAIL_MODIFY_SCOPE],
    )
    return build("gmail", "v1", credentials=credentials, cache_discovery=False)


def authorize_interactively(client_id: str, client_secret: str) -> str:
    """Open a local consent flow and return the refresh token for Railway."""
    from google_auth_oauthlib.flow import InstalledAppFlow

    client_config = {
        "installed": {
            "client_id": client_id,
            "client_secret": client_secret,
            "auth_uri": "https://accounts.google.com/o/oauth2/auth",
            "token_uri": "https://oauth2.googleapis.com/token",
            "redirect_uris": ["http://localhost"],
        }
    }
    flow = InstalledAppFlow.from_client_config(client_config, scopes=[GMAIL_MODIFY_SCOPE])
    credentials = flow.run_local_server(
        host="localhost",
        port=0,
        open_browser=True,
        access_type="offline",
        prompt="consent",
    )
    if not credentials.refresh_token:
        raise RuntimeError("Google did not return a refresh token; revoke prior access and retry")
    return str(credentials.refresh_token)


class GmailAdapter:
    """Gmail alert source and draft publisher. Intentionally exposes no send operation."""

    def __init__(self, service) -> None:
        self.service = service
        # google-api-python-client transports are not thread-safe. Serialize only Gmail
        # calls while allowing profile enrichment and model calls to remain concurrent.
        self._async_lock = asyncio.Lock()

    def _resolve_label_sync(self, profile: ProfileConfig) -> str:
        try:
            response = self.service.users().labels().list(userId="me").execute()
        except Exception as exc:
            if "invalid_scope" in str(exc).casefold():
                raise RuntimeError(
                    "Google rejected the refresh token's OAuth scope. Run "
                    "`newsletter-agent auth`, replace GOOGLE_REFRESH_TOKEN in .env "
                    "with the newly issued token, and retry."
                ) from exc
            raise
        for label in response.get("labels", []):
            if str(label.get("name", "")).casefold() == profile.gmail.label_name.casefold():
                return str(label["id"])
        raise RuntimeError(f"Gmail label not found: {profile.gmail.label_name}")

    def _list_pending_sync(self, context: RunContext, profile: ProfileConfig) -> AlertBatch:
        label_id = self._resolve_label_sync(profile)
        message_refs: list[dict[str, str]] = []
        page_token: str | None = None
        while True:
            kwargs: dict[str, Any] = {
                "userId": "me",
                "labelIds": [label_id],
                "maxResults": 500,
            }
            if page_token:
                kwargs["pageToken"] = page_token
            response = self.service.users().messages().list(**kwargs).execute()
            message_refs.extend(response.get("messages", []))
            page_token = response.get("nextPageToken")
            if not page_token:
                break

        messages: list[SourceMessage] = []
        items = []
        for message_ref in message_refs:
            raw = (
                self.service.users()
                .messages()
                .get(userId="me", id=message_ref["id"], format="full")
                .execute()
            )
            internal_date = gmail_internal_date(raw.get("internalDate"))
            # Start-exclusive and end-inclusive prevents an alert exactly on the
            # previous cutoff from appearing in two consecutive issues.
            if not message_is_in_run_window(context, internal_date):
                continue
            payload = raw.get("payload", {})
            headers = _headers(payload)
            html, text = extract_mime_bodies(payload)
            parsed = parse_scholar_alert(
                html=html,
                text=text,
                message_id=str(raw["id"]),
            )
            messages.append(
                SourceMessage(
                    message_id=str(raw["id"]),
                    thread_id=str(raw.get("threadId")) if raw.get("threadId") else None,
                    internal_date=internal_date,
                    subject=headers.get("subject", "Google Scholar Alert"),
                )
            )
            items.extend(parsed)
        return AlertBatch(messages=tuple(messages), items=tuple(items), label_id=label_id)

    async def list_pending(self, context: RunContext, profile: ProfileConfig) -> AlertBatch:
        async with self._async_lock:
            return await asyncio.to_thread(self._list_pending_sync, context, profile)

    def _acknowledge_sync(self, message_ids: Sequence[str], label_id: str) -> None:
        if not message_ids:
            return
        for start in range(0, len(message_ids), 1000):
            batch = list(message_ids[start : start + 1000])
            (
                self.service.users()
                .messages()
                .batchModify(
                    userId="me",
                    body={"ids": batch, "removeLabelIds": [label_id]},
                )
                .execute()
            )

    async def acknowledge(self, message_ids: Sequence[str], label_id: str) -> None:
        async with self._async_lock:
            await asyncio.to_thread(self._acknowledge_sync, message_ids, label_id)

    def _find_draft_sync(self, subject: str, issue_key: str) -> str | None:
        escaped_subject = subject.replace('"', "")
        response = (
            self.service.users()
            .drafts()
            .list(userId="me", q=f'in:drafts subject:"{escaped_subject}"', maxResults=100)
            .execute()
        )
        for draft in response.get("drafts", []):
            full = (
                self.service.users()
                .drafts()
                .get(userId="me", id=draft["id"], format="metadata")
                .execute()
            )
            headers = _headers(full.get("message", {}).get("payload", {}))
            if headers.get("x-newsletter-agent-issue-key") == issue_key:
                return str(draft["id"])
        return None

    @staticmethod
    def _mime_message(rendered: RenderedNewsletter, issue_key: str) -> str:
        message = EmailMessage()
        message["Subject"] = rendered.subject
        message["Date"] = formatdate(localtime=False)
        message["X-Newsletter-Agent-Issue-Key"] = issue_key
        if rendered.recipients:
            message["To"] = ", ".join(rendered.recipients)
        message.set_content(rendered.text)
        message.add_alternative(rendered.html, subtype="html")
        return base64.urlsafe_b64encode(message.as_bytes()).decode("ascii")

    def _create_or_find_sync(
        self,
        rendered: RenderedNewsletter,
        issue_key: str,
        existing_draft_id: str | None,
    ) -> str:
        if existing_draft_id:
            try:
                self.service.users().drafts().get(userId="me", id=existing_draft_id).execute()
                return existing_draft_id
            except Exception:
                pass
        found = self._find_draft_sync(rendered.subject, issue_key)
        if found:
            return found
        response = (
            self.service.users()
            .drafts()
            .create(
                userId="me",
                body={"message": {"raw": self._mime_message(rendered, issue_key)}},
            )
            .execute()
        )
        return str(response["id"])

    async def create_or_find(
        self,
        rendered: RenderedNewsletter,
        issue_key: str,
        existing_draft_id: str | None = None,
    ) -> str:
        async with self._async_lock:
            return await asyncio.to_thread(
                self._create_or_find_sync,
                rendered,
                issue_key,
                existing_draft_id,
            )


def settings_for_auth() -> tuple[str, str]:
    client_id = os.getenv("GOOGLE_CLIENT_ID")
    client_secret = os.getenv("GOOGLE_CLIENT_SECRET")
    if not client_id or not client_secret:
        raise RuntimeError("GOOGLE_CLIENT_ID and GOOGLE_CLIENT_SECRET are required")
    return client_id, client_secret
