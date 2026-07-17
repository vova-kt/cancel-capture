from __future__ import annotations

import base64
from typing import cast

from openai import AsyncOpenAI
from openai.types.responses.response import Response
from openai.types.responses.response_output_message import ResponseOutputMessage
from openai.types.responses.response_output_text import (
    AnnotationURLCitation,
    ResponseOutputText,
)
from openai.types.responses.tool_param import ToolParam
from pydantic import BaseModel, ConfigDict, Field

from cancel_capture.config import ProviderConfig
from cancel_capture.errors import ConfigurationError, ProviderResponseError
from cancel_capture.models import (
    BilingualText,
    BoundingBox,
    DescribedPhoto,
    Embedding,
    PhotoObservation,
    PreparedImage,
    ProviderIdentity,
    SignDescription,
    SignObservation,
)
from cancel_capture.narrative_models import (
    ClusterTheme,
    NarrativeDraft,
    NarrativeGenerationRequest,
    NewsBrief,
    WebCitation,
)
from cancel_capture.prompts import (
    ARCHIVAL_TEXT_SYSTEM_PROMPT,
    NEWS_SYSTEM_PROMPT,
    VISION_SYSTEM_PROMPT,
    VISION_USER_PROMPT,
    render_archival_text_user_prompt,
    render_news_user_prompt,
)


class _BoxPayload(BaseModel):
    model_config = ConfigDict(extra="forbid")

    left: float = Field(ge=0.0, le=1.0)
    top: float = Field(ge=0.0, le=1.0)
    right: float = Field(ge=0.0, le=1.0)
    bottom: float = Field(ge=0.0, le=1.0)


class _SignObservationPayload(BaseModel):
    model_config = ConfigDict(extra="forbid")

    box: _BoxPayload
    confidence: float = Field(ge=0.0, le=1.0)
    factual_summary: str
    visible_text: list[str]


class _PhotoObservationPayload(BaseModel):
    model_config = ConfigDict(extra="forbid")

    factual_summary: str
    signs: list[_SignObservationPayload]


class _SignDescriptionPayload(BaseModel):
    model_config = ConfigDict(extra="forbid")

    ordinal: int = Field(ge=0)
    description_en: str
    description_ru: str
    topics_en: list[str]
    topics_ru: list[str]


class _PhotoDescriptionPayload(BaseModel):
    model_config = ConfigDict(extra="forbid")

    photo_description_en: str
    photo_description_ru: str
    signs: list[_SignDescriptionPayload]


class _ClusterThemePayload(BaseModel):
    model_config = ConfigDict(extra="forbid")

    title: str
    summary: str


class _NarrativePayload(BaseModel):
    model_config = ConfigDict(extra="forbid")

    title: str
    description: str
    body_markdown: str


def _client(config: ProviderConfig) -> AsyncOpenAI:
    if config.provider != "openai":
        raise ConfigurationError(
            f"Provider {config.provider!r} has no installed adapter; implement the matching Protocol"
        )
    api_key = config.require_api_key()
    if config.base_url is None:
        return AsyncOpenAI(api_key=api_key)
    return AsyncOpenAI(api_key=api_key, base_url=config.base_url)


class OpenAIVisionProvider:
    def __init__(self, config: ProviderConfig) -> None:
        self._config = config
        self._client = _client(config)

    @property
    def identity(self) -> ProviderIdentity:
        return ProviderIdentity(
            provider=self._config.provider,
            model=self._config.model,
            namespace=self._config.identity_namespace,
        )

    async def inspect(self, image: PreparedImage) -> PhotoObservation:
        encoded = base64.b64encode(image.data).decode("ascii")
        response = await self._client.responses.parse(
            model=self._config.model,
            input=[
                {
                    "role": "system",
                    "content": VISION_SYSTEM_PROMPT,
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "input_text",
                            "text": VISION_USER_PROMPT,
                        },
                        {
                            "type": "input_image",
                            "image_url": f"data:{image.media_type};base64,{encoded}",
                            "detail": "high",
                        },
                    ],
                },
            ],
            text_format=_PhotoObservationPayload,
        )
        parsed = response.output_parsed
        if parsed is None:
            raise ProviderResponseError("Vision provider returned no structured observation")
        signs: list[SignObservation] = []
        for ordinal, sign in enumerate(parsed.signs):
            try:
                box = BoundingBox(
                    left=sign.box.left,
                    top=sign.box.top,
                    right=sign.box.right,
                    bottom=sign.box.bottom,
                )
            except ValueError as error:
                raise ProviderResponseError(
                    "Vision provider returned an invalid bounding box"
                ) from error
            signs.append(
                SignObservation(
                    ordinal=ordinal,
                    box=box,
                    confidence=sign.confidence,
                    factual_summary=sign.factual_summary,
                    visible_text=tuple(text.strip() for text in sign.visible_text if text.strip()),
                )
            )
        return PhotoObservation(factual_summary=parsed.factual_summary, signs=tuple(signs))


class OpenAITextProvider:
    def __init__(self, config: ProviderConfig) -> None:
        self._config = config
        self._client = _client(config)

    @property
    def identity(self) -> ProviderIdentity:
        return ProviderIdentity(
            provider=self._config.provider,
            model=self._config.model,
            namespace=self._config.identity_namespace,
        )

    async def describe(self, observation: PhotoObservation) -> DescribedPhoto:
        payload = {
            "photo": observation.factual_summary,
            "signs": [
                {
                    "ordinal": sign.ordinal,
                    "observation": sign.factual_summary,
                    "visible_text": list(sign.visible_text),
                }
                for sign in observation.signs
            ],
        }
        response = await self._client.responses.parse(
            model=self._config.model,
            input=[
                {
                    "role": "system",
                    "content": ARCHIVAL_TEXT_SYSTEM_PROMPT,
                },
                {
                    "role": "user",
                    "content": render_archival_text_user_prompt(payload),
                },
            ],
            text_format=_PhotoDescriptionPayload,
        )
        parsed = response.output_parsed
        if parsed is None:
            raise ProviderResponseError("Text provider returned no structured description")
        descriptions = tuple(
            SignDescription(
                ordinal=sign.ordinal,
                text=BilingualText(en=sign.description_en, ru=sign.description_ru),
                topics_en=tuple(topic.strip() for topic in sign.topics_en if topic.strip()),
                topics_ru=tuple(topic.strip() for topic in sign.topics_ru if topic.strip()),
            )
            for sign in sorted(parsed.signs, key=lambda item: item.ordinal)
        )
        if len(descriptions) != len(observation.signs) or tuple(
            description.ordinal for description in descriptions
        ) != tuple(range(len(observation.signs))):
            raise ProviderResponseError("Text provider changed or omitted sign ordinals")
        return DescribedPhoto(
            photo=BilingualText(
                en=parsed.photo_description_en,
                ru=parsed.photo_description_ru,
            ),
            signs=descriptions,
        )


class OpenAIEmbeddingProvider:
    def __init__(self, config: ProviderConfig) -> None:
        self._config = config
        self._client = _client(config)

    @property
    def identity(self) -> ProviderIdentity:
        return ProviderIdentity(
            provider=self._config.provider,
            model=self._config.model,
            namespace=self._config.identity_namespace,
        )

    async def embed(self, texts: tuple[str, ...]) -> tuple[Embedding, ...]:
        if not texts:
            return ()
        if self._config.dimensions is None:
            response = await self._client.embeddings.create(
                model=self._config.model,
                input=list(texts),
                encoding_format="float",
            )
        else:
            response = await self._client.embeddings.create(
                model=self._config.model,
                input=list(texts),
                encoding_format="float",
                dimensions=self._config.dimensions,
            )
        ordered = sorted(response.data, key=lambda item: item.index)
        if len(ordered) != len(texts):
            raise ProviderResponseError("Embedding provider returned the wrong vector count")
        if tuple(item.index for item in ordered) != tuple(range(len(texts))):
            raise ProviderResponseError("Embedding provider returned invalid vector indexes")
        return tuple(
            Embedding(
                identity=self.identity, values=tuple(float(value) for value in item.embedding)
            )
            for item in ordered
        )


class OpenAINarrativeProvider:
    def __init__(self, config: ProviderConfig) -> None:
        self._config = config
        self._client = _client(config)

    @property
    def identity(self) -> ProviderIdentity:
        return ProviderIdentity(
            provider=self._config.provider,
            model=self._config.model,
            namespace=self._config.identity_namespace,
        )

    async def generate(self, request: NarrativeGenerationRequest) -> NarrativeDraft:
        response = await self._client.responses.parse(
            model=self._config.model,
            input=[
                {"role": "system", "content": request.system_prompt},
                {"role": "user", "content": request.user_prompt},
            ],
            text_format=_NarrativePayload,
        )
        parsed = response.output_parsed
        if parsed is None:
            raise ProviderResponseError("Narrative provider returned no structured story")
        try:
            return NarrativeDraft(
                title=parsed.title.strip(),
                description=parsed.description.strip(),
                body_markdown=parsed.body_markdown.strip(),
            )
        except ValueError as error:
            raise ProviderResponseError(str(error)) from error


class OpenAIClusterThemeProvider:
    def __init__(self, config: ProviderConfig) -> None:
        self._config = config
        self._client = _client(config)

    @property
    def identity(self) -> ProviderIdentity:
        return ProviderIdentity(
            provider=self._config.provider,
            model=self._config.model,
            namespace=self._config.identity_namespace,
        )

    async def summarize(self, system_prompt: str, user_prompt: str) -> ClusterTheme:
        response = await self._client.responses.parse(
            model=self._config.model,
            input=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            text_format=_ClusterThemePayload,
        )
        parsed = response.output_parsed
        if parsed is None:
            raise ProviderResponseError("Cluster theme provider returned no structured summary")
        try:
            return ClusterTheme(title=parsed.title.strip(), summary=parsed.summary.strip())
        except ValueError as error:
            raise ProviderResponseError(str(error)) from error


class OpenAICurrentNewsProvider:
    def __init__(self, config: ProviderConfig) -> None:
        self._config = config
        self._client = _client(config)

    @property
    def identity(self) -> ProviderIdentity:
        return ProviderIdentity(
            provider=self._config.provider,
            model=self._config.model,
            namespace=self._config.identity_namespace,
        )

    async def research(self, query: str, *, current_date: str) -> NewsBrief:
        tool: ToolParam = cast(
            ToolParam,
            {"type": "web_search", "search_context_size": "medium"},
        )
        response = await self._client.responses.create(
            model=self._config.model,
            tools=[tool],
            input=[
                {"role": "system", "content": NEWS_SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": render_news_user_prompt(query, current_date=current_date),
                },
            ],
        )
        return _extract_news_brief(response)


def _extract_news_brief(response: Response) -> NewsBrief:
    parts: list[str] = []
    citations: dict[tuple[str, str], WebCitation] = {}
    for item in response.output:
        if not isinstance(item, ResponseOutputMessage):
            continue
        for content in item.content:
            if not isinstance(content, ResponseOutputText):
                continue
            parts.append(content.text)
            for annotation in content.annotations:
                if not isinstance(annotation, AnnotationURLCitation):
                    continue
                title = annotation.title.strip() or annotation.url
                url = annotation.url.strip()
                if not url.startswith(("http://", "https://")):
                    continue
                key = (title, url)
                if key in citations:
                    continue
                try:
                    citations[key] = WebCitation(title=title, url=url)
                except ValueError:
                    continue
    markdown = "\n\n".join(part.strip() for part in parts if part.strip())
    if not markdown:
        raise ProviderResponseError("Current-news provider returned no text output")
    try:
        return NewsBrief(markdown=markdown, citations=tuple(citations.values()))
    except ValueError as error:
        raise ProviderResponseError(str(error)) from error
