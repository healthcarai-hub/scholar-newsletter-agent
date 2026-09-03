import asyncio

from newsletter_agent.config import AppConfig
from newsletter_agent.domain.models import EnrichedItem, ResearchItem
from newsletter_agent.infrastructure.llm import (
    OpenAICompatibleCategorizer,
    OpenAICompatibleEmbeddingProvider,
    _retry_after_seconds,
    render_prompt,
    valid_issue_summary,
)


def test_topic_and_categories_are_interpolated():
    prompt = render_prompt(
        "Summarize {topic} using {categories}",
        topic="migraine research",
        categories="Mechanisms, Treatments",
    )
    assert prompt == "Summarize migraine research using Mechanisms, Treatments"


def test_issue_summary_contract_requires_three_to_four_sentences_and_sixty_to_ninety_words():
    valid = (
        "This week's research examines molecular pathways that may shape migraine susceptibility and symptom expression across different patient groups. "
        "Several studies also evaluate treatment response, safety, and clinical outcomes using observational and controlled designs. "
        "Diagnostic work explores practical biomarkers and monitoring approaches that could support more precise assessment in future studies. "
        "Together, the results show a broad evidence base while leaving important questions for replication and clinical validation."
    )
    assert valid_issue_summary(valid)
    assert not valid_issue_summary("Too short. Only two sentences.")


class _Response:
    def __init__(self, retry_after: str | None):
        self.headers = {"retry-after": retry_after} if retry_after else {}


def test_retry_after_header_is_honored_and_capped():
    assert _retry_after_seconds(_Response("12.5"), fallback=2, maximum=60) == 12.5
    assert _retry_after_seconds(_Response("120"), fallback=2, maximum=60) == 60
    assert _retry_after_seconds(_Response(None), fallback=2, maximum=60) == 2


def test_classification_and_item_summary_share_one_model_call(config_dict):
    profile = AppConfig.model_validate(config_dict).profiles["migraine"]
    original = ResearchItem(
        title="A migraine paper",
        url="https://example.org/paper",
        snippet="A grounded Scholar snippet.",
        source_message_id="message-1",
    )
    enriched = EnrichedItem(
        item=original,
        canonical_url=original.url,
        fingerprint="fingerprint",
        source_text="Title: A migraine paper\nScholar snippet: A grounded Scholar snippet.",
    )
    categorizer = OpenAICompatibleCategorizer(
        base_url="https://example.invalid/v1",
        api_key="test",
        model="test-model",
    )
    calls = []

    async def completion(prompt, response_contract):
        calls.append((prompt, response_contract))
        return {
            "category": "Mechanisms",
            "headline": "A migraine paper",
            "brief": "A grounded Scholar snippet.",
        }

    categorizer._completion = completion
    result = asyncio.run(categorizer.classify(enriched, profile))

    assert result.category == "Mechanisms"
    assert len(calls) == 1
    assert calls[0][0].count(enriched.source_text) == 1
    assert "Do not include DOI" in calls[0][0]
    assert result.source_label is None


def test_embedding_input_is_bounded_without_truncating_stored_source(monkeypatch):
    import httpx

    captured = {}

    class Response:
        status_code = 200
        headers = {"content-type": "application/json"}

        def raise_for_status(self):
            return None

        def json(self):
            return {"data": [{"index": 0, "embedding": [0.1, 0.2]}]}

    class Client:
        def __init__(self, **kwargs):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            return None

        async def post(self, url, *, headers, json):
            captured["payload"] = json
            return Response()

    monkeypatch.setattr(httpx, "AsyncClient", Client)
    provider = OpenAICompatibleEmbeddingProvider(
        base_url="https://example.invalid/v1",
        api_key="test",
        model="embedding-model",
        max_input_chars=5,
    )
    original = "abcdefghij"

    vectors = asyncio.run(provider.embed([original]))

    assert captured["payload"]["input"] == ["abcde"]
    assert original == "abcdefghij"
    assert vectors == [[0.1, 0.2]]
