from __future__ import annotations

import os
from pathlib import Path
from string import Formatter
from typing import Any

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator, model_validator

from newsletter_agent.domain.models import FALLBACK_CATEGORY


ALLOWED_PROMPT_FIELDS = {"topic", "categories", "source_text", "items", "issue_date", "result_count"}
PROMPT_REQUIREMENTS = {
    "categorization": {"topic", "categories", "source_text"},
    "item_summary": {"topic", "source_text"},
    "issue_summary": {"topic", "categories", "items"},
}
NEWSLETTER_ALLOWED_FIELDS = {"topic", "issue_date", "result_count", "profile_id"}


def template_fields(template: str) -> set[str]:
    fields: set[str] = set()
    try:
        for _, name, _, _ in Formatter().parse(template):
            if name:
                fields.add(name)
    except ValueError as exc:
        raise ValueError(f"invalid template syntax: {exc}") from exc
    return fields


def validate_template(
    template: str,
    *,
    allowed: set[str],
    required: set[str] | None = None,
    label: str,
) -> str:
    fields = template_fields(template)
    unknown = fields - allowed
    missing = (required or set()) - fields
    if unknown:
        raise ValueError(f"{label} contains unsupported placeholders: {sorted(unknown)}")
    if missing:
        raise ValueError(f"{label} is missing required placeholders: {sorted(missing)}")
    return template


class CategoryConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    name: str = Field(min_length=1, max_length=120)
    guidance: str = Field(min_length=1, max_length=600)

    @field_validator("name", "guidance")
    @classmethod
    def strip_text(cls, value: str) -> str:
        return value.strip()


class GmailProfileConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    label_name: str = Field(min_length=1)

    @field_validator("label_name")
    @classmethod
    def strip_label_name(cls, value: str) -> str:
        return value.strip()


class PromptConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    categorization: str
    item_summary: str
    issue_summary: str

    @model_validator(mode="after")
    def validate_prompts(self) -> "PromptConfig":
        for field_name, required in PROMPT_REQUIREMENTS.items():
            validate_template(
                getattr(self, field_name),
                allowed=ALLOWED_PROMPT_FIELDS,
                required=required,
                label=f"prompts.{field_name}",
            )
        return self


class SignatureConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    name: str = Field(min_length=1, max_length=160)
    title: str | None = Field(default=None, max_length=160)
    email: str | None = Field(default=None, max_length=320)
    phone: str | None = Field(default=None, max_length=80)
    disclaimer: str | None = Field(default=None, max_length=800)

    @field_validator("name", "title", "email", "phone", "disclaimer")
    @classmethod
    def strip_signature_text(cls, value: str | None) -> str | None:
        return value.strip() if value else value


class NewsletterConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    subject_template: str
    title_template: str
    recipients: tuple[str, ...] = ()
    subtitle: str = "Curated insights from recent publications"
    accent_color: str = "#155eef"
    greeting: str = "Hello,"
    introduction: str = (
        "Here is this week's research wrap-up, with the most relevant publications "
        "organized by category."
    )
    results_lead: str = "Here are the latest publications:"
    closing: str = "Have a great week."
    signature: SignatureConfig | None = None
    footer: str = "Research newsletter prepared for review."

    @field_validator("subject_template", "title_template")
    @classmethod
    def validate_newsletter_template(cls, value: str) -> str:
        return validate_template(
            value,
            allowed=NEWSLETTER_ALLOWED_FIELDS,
            required={"topic"},
            label="newsletter template",
        )

    @field_validator("accent_color")
    @classmethod
    def validate_color(cls, value: str) -> str:
        value = value.strip()
        if len(value) != 7 or not value.startswith("#"):
            raise ValueError("accent_color must be a six-digit hex color")
        int(value[1:], 16)
        return value.lower()


class ProfileConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    enabled: bool = True
    rag_indexing_enabled: bool = True
    topic: str = Field(min_length=1, max_length=160)
    gmail: GmailProfileConfig
    categories: tuple[CategoryConfig, ...] = Field(min_length=1)
    prompts: PromptConfig
    newsletter: NewsletterConfig

    @field_validator("topic")
    @classmethod
    def strip_topic(cls, value: str) -> str:
        return value.strip()

    @model_validator(mode="after")
    def unique_categories(self) -> "ProfileConfig":
        names = [category.name.casefold() for category in self.categories]
        if len(names) != len(set(names)):
            raise ValueError("category names must be unique within a profile")
        if FALLBACK_CATEGORY.casefold() in names:
            raise ValueError(f"{FALLBACK_CATEGORY!r} is reserved as the automatic fallback")
        return self

    def category_prompt(self) -> str:
        lines = [f"- {category.name}: {category.guidance}" for category in self.categories]
        lines.append(f"- {FALLBACK_CATEGORY}: use only when no defined category fits.")
        return "\n".join(lines)

    @property
    def allowed_categories(self) -> tuple[str, ...]:
        return tuple(category.name for category in self.categories) + (FALLBACK_CATEGORY,)


class RuntimeConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    timezone: str = "Asia/Kolkata"
    workspace_id: str = "default"
    alert_lookback_days: int = Field(default=7, ge=1, le=90)
    max_profile_concurrency: int = Field(default=2, ge=1, le=20)
    max_enrichment_concurrency: int = Field(default=6, ge=1, le=50)
    max_llm_concurrency: int = Field(default=3, ge=1, le=20)
    llm_request_interval_seconds: float = Field(default=0.0, ge=0, le=60)
    llm_max_retries: int = Field(default=6, ge=0, le=12)
    llm_max_retry_wait_seconds: float = Field(default=60.0, ge=1, le=300)
    embedding_batch_size: int = Field(default=32, ge=1, le=256)
    embedding_max_input_chars: int = Field(default=8_000, ge=256, le=100_000)
    request_timeout_seconds: float = Field(default=15.0, ge=1, le=120)
    max_page_bytes: int = Field(default=2_000_000, ge=10_000, le=10_000_000)
    preview_directory: str = "outputs/previews"


class AppConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    runtime: RuntimeConfig = RuntimeConfig()
    profiles: dict[str, ProfileConfig]

    @model_validator(mode="after")
    def validate_profiles(self) -> "AppConfig":
        if not self.profiles:
            raise ValueError("at least one profile is required")
        normalized_ids: set[str] = set()
        active_labels: dict[str, str] = {}
        for profile_id, profile in self.profiles.items():
            clean_id = profile_id.strip()
            if clean_id != profile_id or not clean_id:
                raise ValueError("profile IDs must be non-empty and may not have surrounding whitespace")
            folded = clean_id.casefold()
            if folded in normalized_ids:
                raise ValueError(f"duplicate profile ID: {profile_id}")
            normalized_ids.add(folded)
            if profile.enabled:
                label_key = profile.gmail.label_name.casefold()
                if label_key in active_labels:
                    raise ValueError(
                        f"enabled profiles {active_labels[label_key]!r} and {profile_id!r} "
                        "use the same Gmail label"
                    )
                active_labels[label_key] = profile_id
        return self


class EnvironmentSettings(BaseModel):
    model_config = ConfigDict(extra="ignore", frozen=True)
    database_url: str | None = None
    google_client_id: str | None = None
    google_client_secret: str | None = None
    google_refresh_token: str | None = None
    llm_base_url: str | None = None
    llm_api_key: str | None = None
    llm_model: str | None = None
    embedding_base_url: str | None = None
    embedding_api_key: str | None = None
    embedding_model: str | None = None
    embedding_dimensions: int | None = None

    @classmethod
    def from_env(cls) -> "EnvironmentSettings":
        return cls(
            database_url=os.getenv("DATABASE_URL"),
            google_client_id=os.getenv("GOOGLE_CLIENT_ID"),
            google_client_secret=os.getenv("GOOGLE_CLIENT_SECRET"),
            google_refresh_token=os.getenv("GOOGLE_REFRESH_TOKEN"),
            llm_base_url=os.getenv("LLM_BASE_URL"),
            llm_api_key=os.getenv("LLM_API_KEY"),
            llm_model=os.getenv("LLM_MODEL"),
            embedding_base_url=os.getenv("EMBEDDING_BASE_URL"),
            embedding_api_key=os.getenv("EMBEDDING_API_KEY"),
            embedding_model=os.getenv("EMBEDDING_MODEL"),
            embedding_dimensions=(
                int(os.environ["EMBEDDING_DIMENSIONS"])
                if os.getenv("EMBEDDING_DIMENSIONS")
                else None
            ),
        )


def load_config(path: str | Path) -> AppConfig:
    config_path = Path(path)
    try:
        raw: Any = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise ValueError(f"configuration file not found: {config_path}") from exc
    except yaml.YAMLError as exc:
        raise ValueError(f"invalid YAML in {config_path}: {exc}") from exc
    try:
        return AppConfig.model_validate(raw)
    except ValidationError as exc:
        raise ValueError(str(exc)) from exc
