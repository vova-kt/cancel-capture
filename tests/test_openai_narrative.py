from __future__ import annotations

from types import SimpleNamespace
from typing import cast

import pytest
from openai.types.responses.response_output_message import ResponseOutputMessage
from openai.types.responses.response_output_text import (
    AnnotationURLCitation,
    ResponseOutputText,
)

from cancel_capture.adapters.openai_provider import (
    OpenAIClusterThemeProvider,
    OpenAICurrentNewsProvider,
    OpenAINarrativeProvider,
    _extract_news_brief,  # pyright: ignore [reportPrivateUsage]
)
from cancel_capture.config import ProviderConfig
from cancel_capture.errors import ProviderResponseError
from cancel_capture.narrative_models import (
    NarrativeGenerationRequest,
    NarrativeLanguage,
    NarrativeSource,
    NewsBrief,
)

CONFIG = ProviderConfig(
    provider="openai",
    api_key="unused",
    base_url=None,
    model="test-model",
    identity_namespace="test-namespace",
)


class _Responses:
    def __init__(self, client: FakeResponsesClient) -> None:
        self._client = client

    async def parse(self, **kwargs: object) -> object:
        self._client.parse_requests.append(kwargs)
        return SimpleNamespace(output_parsed=self._client.payload)

    async def create(self, **kwargs: object) -> object:
        self._client.create_requests.append(kwargs)
        return self._client.payload


class FakeResponsesClient:
    def __init__(self, payload: object) -> None:
        self.payload = payload
        self.parse_requests: list[dict[str, object]] = []
        self.create_requests: list[dict[str, object]] = []

    @property
    def responses(self) -> _Responses:
        return _Responses(self)


def _narrative_request() -> NarrativeGenerationRequest:
    return NarrativeGenerationRequest(
        start_date="2026-07-17",
        end_year=2031,
        language=NarrativeLanguage.ENGLISH,
        reading_minutes=2,
        target_words=480,
        strategy="family_chronicle",
        system_prompt="System",
        user_prompt="User",
        news=NewsBrief(markdown="- Something.", citations=()),
        sources=(
            NarrativeSource(
                sign_id="anchor",
                description="anchor description",
                topics=("topic",),
                weight=2.5,
                similarity_to_anchor=1.0,
                is_anchor=True,
            ),
        ),
    )


def _install(provider: object, client: FakeResponsesClient) -> None:
    provider._client = client


async def test_narrative_provider_returns_structured_draft_from_parsed_response() -> None:
    payload = SimpleNamespace(
        title="A title",
        description="A description",
        body_markdown="# Body\n\nContent.",
    )
    client = FakeResponsesClient(payload)
    provider = object.__new__(OpenAINarrativeProvider)
    provider._config = CONFIG
    _install(provider, client)

    draft = await provider.generate(_narrative_request())

    assert draft.title == "A title"
    assert draft.description == "A description"
    assert draft.body_markdown == "# Body\n\nContent."
    call = client.parse_requests[0]
    inputs = cast(list[dict[str, object]], call["input"])
    assert inputs[0]["role"] == "system"
    assert inputs[0]["content"] == "System"


async def test_narrative_provider_raises_when_no_parsed_payload() -> None:
    provider = object.__new__(OpenAINarrativeProvider)
    provider._config = CONFIG
    _install(provider, FakeResponsesClient(None))

    with pytest.raises(ProviderResponseError):
        await provider.generate(_narrative_request())


async def test_cluster_theme_provider_returns_stripped_title_and_summary() -> None:
    payload = SimpleNamespace(title=" A theme ", summary=" A summary. ")
    provider = object.__new__(OpenAIClusterThemeProvider)
    provider._config = CONFIG
    client = FakeResponsesClient(payload)
    _install(provider, client)

    theme = await provider.summarize("system", "user")

    assert theme.title == "A theme"
    assert theme.summary == "A summary."
    call = client.parse_requests[0]
    inputs = cast(list[dict[str, object]], call["input"])
    assert inputs[0]["content"] == "system"
    assert inputs[1]["content"] == "user"


def test_extract_news_brief_collects_text_and_citations() -> None:
    text = ResponseOutputText(
        type="output_text",
        text="- Example story",
        annotations=[
            AnnotationURLCitation(
                type="url_citation",
                title="Example",
                url="https://example.com/article",
                start_index=0,
                end_index=10,
            )
        ],
    )
    message = ResponseOutputMessage(
        id="msg-1",
        type="message",
        role="assistant",
        status="completed",
        content=[text],
    )
    response = SimpleNamespace(output=[message])

    brief = _extract_news_brief(cast("object", response))  # type: ignore[arg-type]

    assert isinstance(brief, NewsBrief)
    assert brief.markdown == "- Example story"
    assert brief.citations[0].url == "https://example.com/article"


def test_extract_news_brief_raises_when_empty() -> None:
    empty_text = ResponseOutputText(type="output_text", text="", annotations=[])
    message = ResponseOutputMessage(
        id="msg-2",
        type="message",
        role="assistant",
        status="completed",
        content=[empty_text],
    )
    response = SimpleNamespace(output=[message])

    with pytest.raises(ProviderResponseError):
        _extract_news_brief(cast("object", response))  # type: ignore[arg-type]


async def test_current_news_provider_wires_up_web_search_tool() -> None:
    text = ResponseOutputText(type="output_text", text="- One event", annotations=[])
    message = ResponseOutputMessage(
        id="msg-1",
        type="message",
        role="assistant",
        status="completed",
        content=[text],
    )
    response = SimpleNamespace(output=[message])

    provider = object.__new__(OpenAICurrentNewsProvider)
    provider._config = CONFIG
    client = FakeResponsesClient(response)
    _install(provider, client)

    brief = await provider.research("prohibitions", current_date="2026-07-17")
    assert brief.markdown == "- One event"

    call = client.create_requests[0]
    tools = cast(list[dict[str, object]], call["tools"])
    assert tools[0]["type"] == "web_search"
