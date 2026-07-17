from __future__ import annotations

import base64
import json

from openai import AsyncOpenAI
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
                    "content": (
                        "Inspect documentary photographs for prohibition signs. A qualifying sign is "
                        "round, has a red border, and visibly prohibits something with a red diagonal "
                        "slash or crossing. Return every distinct qualifying sign, including partial "
                        "but recognizable signs. Do not return ordinary traffic signs, red circles "
                        "without a prohibition mark, logos, or decorative circles. Bounding boxes use "
                        "normalized image coordinates from 0 to 1. Describe only visible facts; do not "
                        "invent location, intent, or text. Order signs top-to-bottom then left-to-right."
                    ),
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "input_text",
                            "text": (
                                "Describe the complete scene factually and locate every prohibition "
                                "sign. Transcribe visible sign text exactly when legible."
                            ),
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
                    "content": (
                        "Write accurate archival descriptions in both English and Russian. The full "
                        "photo description must cover setting, objects, people, composition, and the "
                        "relationship of signs to the scene without guessing hidden facts. Each sign "
                        "description must explain exactly what appears prohibited, its pictogram, "
                        "wording, condition, and visual context. Preserve quoted text. Produce concise "
                        "topic tags in both languages that will make semantic retrieval useful. Do not "
                        "write a narrative about society or infer motivations. Keep sign ordinals "
                        "unchanged."
                    ),
                },
                {
                    "role": "user",
                    "content": json.dumps(payload, ensure_ascii=False, separators=(",", ":")),
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
