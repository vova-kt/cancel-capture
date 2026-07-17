from __future__ import annotations

import math
import re
from dataclasses import dataclass
from datetime import UTC, datetime
from enum import StrEnum
from uuid import uuid4

_IDENTIFIER = re.compile(r"[A-Za-z0-9][A-Za-z0-9_-]{0,127}\Z")


def validate_narrative_identifier(value: str, label: str) -> None:
    if _IDENTIFIER.fullmatch(value) is None:
        raise ValueError(f"{label} is not a safe identifier")


class NarrativeLanguage(StrEnum):
    ENGLISH = "English"
    RUSSIAN = "Russian"


@dataclass(frozen=True, slots=True)
class WebCitation:
    title: str
    url: str

    def __post_init__(self) -> None:
        if not self.title.strip() or not self.url.startswith(("https://", "http://")):
            raise ValueError("A web citation requires a title and an HTTP(S) URL")


@dataclass(frozen=True, slots=True)
class NewsBrief:
    markdown: str
    citations: tuple[WebCitation, ...]

    def __post_init__(self) -> None:
        if not self.markdown.strip():
            raise ValueError("A current-news brief cannot be empty")


@dataclass(frozen=True, slots=True)
class NarrativeSource:
    sign_id: str
    description: str
    topics: tuple[str, ...]
    weight: float
    similarity_to_anchor: float
    is_anchor: bool


@dataclass(frozen=True, slots=True)
class NarrativeGenerationRequest:
    start_date: str
    end_year: int
    language: NarrativeLanguage
    reading_minutes: int
    target_words: int
    strategy: str
    system_prompt: str
    user_prompt: str
    news: NewsBrief
    sources: tuple[NarrativeSource, ...]

    def __post_init__(self) -> None:
        if self.reading_minutes <= 0 or self.target_words <= 0:
            raise ValueError("Narrative length must be positive")
        if not self.sources or not self.sources[0].is_anchor:
            raise ValueError("The first narrative source must be the anchor")


@dataclass(frozen=True, slots=True)
class NarrativeDraft:
    title: str
    description: str
    body_markdown: str

    def __post_init__(self) -> None:
        if not self.title.strip() or "\n" in self.title or "\r" in self.title:
            raise ValueError("Narrative title must be a non-empty single line")
        if not self.description.strip() or not self.body_markdown.strip():
            raise ValueError("Narrative description and body cannot be empty")


@dataclass(frozen=True, slots=True)
class ClusterTheme:
    title: str
    summary: str

    def __post_init__(self) -> None:
        if not self.title.strip() or not self.summary.strip():
            raise ValueError("Cluster theme title and summary cannot be empty")


@dataclass(frozen=True, slots=True)
class NarrativeArtifactMetadata:
    attempt_id: str
    created_at: str
    anchor_sign_id: str
    source_sign_ids: tuple[str, ...]
    strategy: str
    language: str
    reading_minutes: int
    similarity_mode: str
    similarity_threshold: float
    semantic_weight: float
    random_seed: int | None
    provider: str
    model: str
    provider_namespace: str
    system_prompt: str
    user_prompt: str
    web_citations: tuple[WebCitation, ...] = ()
    output_version: int = 1

    def __post_init__(self) -> None:
        self._validate_identifier(self.attempt_id, "Narrative attempt ID")
        self._validate_identifier(self.anchor_sign_id, "Anchor sign ID")
        self._parse_timestamp(self.created_at)
        if not self.source_sign_ids:
            raise ValueError("A narrative requires at least one source sign")
        if len(set(self.source_sign_ids)) != len(self.source_sign_ids):
            raise ValueError("Narrative source sign IDs must be unique")
        for source_sign_id in self.source_sign_ids:
            self._validate_identifier(source_sign_id, "Source sign ID")
        if self.anchor_sign_id in self.source_sign_ids:
            raise ValueError("The anchor sign cannot also be a source sign")
        if (
            not self.strategy.strip()
            or not self.language.strip()
            or not self.similarity_mode.strip()
        ):
            raise ValueError("Narrative strategy, language, and similarity mode are required")
        if self.reading_minutes <= 0:
            raise ValueError("Narrative reading time must be positive")
        threshold = self.similarity_threshold
        if not math.isfinite(threshold) or not -1.0 <= threshold <= 1.0:
            raise ValueError("Narrative similarity threshold must be within [-1, 1]")
        if not math.isfinite(self.semantic_weight) or not 0.0 <= self.semantic_weight <= 1.0:
            raise ValueError("Narrative semantic weight must be within [0, 1]")
        if isinstance(self.random_seed, bool):
            raise ValueError("Narrative random seed must be an integer or null")
        for value, label in (
            (self.provider, "Narrative provider"),
            (self.model, "Narrative model"),
            (self.provider_namespace, "Narrative provider namespace"),
            (self.system_prompt, "Narrative system prompt"),
            (self.user_prompt, "Narrative user prompt"),
        ):
            if not value.strip():
                raise ValueError(f"{label} cannot be empty")
        if self.output_version <= 0:
            raise ValueError("Narrative output version must be positive")

    @classmethod
    def create(
        cls,
        *,
        anchor_sign_id: str,
        source_sign_ids: tuple[str, ...],
        strategy: str,
        language: str,
        reading_minutes: int,
        similarity_mode: str,
        similarity_threshold: float,
        semantic_weight: float,
        random_seed: int | None,
        provider: str,
        model: str,
        provider_namespace: str,
        system_prompt: str,
        user_prompt: str,
        web_citations: tuple[WebCitation, ...] = (),
        output_version: int = 1,
        attempt_id: str | None = None,
        created_at: datetime | None = None,
    ) -> NarrativeArtifactMetadata:
        timestamp = created_at or datetime.now(UTC)
        if timestamp.tzinfo is None or timestamp.utcoffset() is None:
            raise ValueError("Narrative creation time must include a timezone")
        return cls(
            attempt_id=attempt_id or uuid4().hex,
            created_at=timestamp.isoformat(),
            anchor_sign_id=anchor_sign_id,
            source_sign_ids=source_sign_ids,
            strategy=strategy,
            language=language,
            reading_minutes=reading_minutes,
            similarity_mode=similarity_mode,
            similarity_threshold=similarity_threshold,
            semantic_weight=semantic_weight,
            random_seed=random_seed,
            provider=provider,
            model=model,
            provider_namespace=provider_namespace,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            web_citations=web_citations,
            output_version=output_version,
        )

    @staticmethod
    def _validate_identifier(value: str, label: str) -> None:
        validate_narrative_identifier(value, label)

    @staticmethod
    def _parse_timestamp(value: str) -> datetime:
        try:
            parsed = datetime.fromisoformat(value)
        except ValueError as error:
            raise ValueError("Narrative creation time must be ISO 8601") from error
        if parsed.tzinfo is None or parsed.utcoffset() is None:
            raise ValueError("Narrative creation time must include a timezone")
        return parsed


@dataclass(frozen=True, slots=True)
class NarrativeArtifact:
    title: str
    description: str
    body_markdown: str
    metadata: NarrativeArtifactMetadata

    def __post_init__(self) -> None:
        if not self.title.strip() or "\n" in self.title or "\r" in self.title:
            raise ValueError("Narrative title must fit on one non-empty line")
        if len(self.title) > 200 or not self.description.strip() or len(self.description) > 2_000:
            raise ValueError("Narrative title or description has an invalid length")
        if not self.body_markdown.strip():
            raise ValueError("Narrative body cannot be empty")


@dataclass(frozen=True, slots=True)
class StoredNarrativeArtifact:
    relative_path: str
    artifact: NarrativeArtifact
