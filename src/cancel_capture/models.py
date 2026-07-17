from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from math import isfinite


class ItemKind(StrEnum):
    PHOTO = "photo"
    SIGN = "sign"


class EmbeddingKind(StrEnum):
    SEMANTIC = "semantic"
    VISUAL = "visual"


class SourceKind(StrEnum):
    BOT = "bot"
    CHANNEL_IMPORT = "channel_import"
    STREAMLIT = "streamlit"


class ReviewStatus(StrEnum):
    READY = "ready"
    PENDING = "pending_review"
    PUBLISHING = "publishing"
    PUBLISHED = "published"
    REJECTED = "rejected"
    FAILED = "failed"


class TelegramMessageRole(StrEnum):
    INBOUND = "inbound"
    HISTORY = "history"
    PREVIEW = "preview"
    CHANNEL_POST = "channel_post"


@dataclass(frozen=True, slots=True)
class BoundingBox:
    left: float
    top: float
    right: float
    bottom: float

    def __post_init__(self) -> None:
        values = (self.left, self.top, self.right, self.bottom)
        if not all(isfinite(value) for value in values):
            raise ValueError("Bounding-box coordinates must be finite")
        if not all(0.0 <= value <= 1.0 for value in values):
            raise ValueError("Bounding-box coordinates must be normalized to [0, 1]")
        if self.left >= self.right or self.top >= self.bottom:
            raise ValueError("Bounding box must have positive width and height")

    @classmethod
    def full_frame(cls) -> BoundingBox:
        return cls(left=0.0, top=0.0, right=1.0, bottom=1.0)

    def expanded(self, ratio: float) -> BoundingBox:
        if ratio < 0:
            raise ValueError("Expansion ratio cannot be negative")
        horizontal = (self.right - self.left) * ratio
        vertical = (self.bottom - self.top) * ratio
        return BoundingBox(
            left=max(0.0, self.left - horizontal),
            top=max(0.0, self.top - vertical),
            right=min(1.0, self.right + horizontal),
            bottom=min(1.0, self.bottom + vertical),
        )


@dataclass(frozen=True, slots=True)
class BilingualText:
    en: str
    ru: str

    def __post_init__(self) -> None:
        if not self.en.strip() or not self.ru.strip():
            raise ValueError("Both English and Russian text are required")

    def search_text(self) -> str:
        return f"English:\n{self.en.strip()}\n\nRussian:\n{self.ru.strip()}"


@dataclass(frozen=True, slots=True)
class SignObservation:
    ordinal: int
    box: BoundingBox
    confidence: float
    factual_summary: str
    visible_text: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.ordinal < 0:
            raise ValueError("Sign ordinal cannot be negative")
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError("Confidence must be normalized to [0, 1]")
        if not self.factual_summary.strip():
            raise ValueError("A sign observation requires a factual summary")


@dataclass(frozen=True, slots=True)
class PhotoObservation:
    factual_summary: str
    signs: tuple[SignObservation, ...]

    def __post_init__(self) -> None:
        if not self.factual_summary.strip():
            raise ValueError("A photo observation requires a factual summary")
        expected = tuple(range(len(self.signs)))
        actual = tuple(sign.ordinal for sign in self.signs)
        if actual != expected:
            raise ValueError("Sign ordinals must be contiguous and zero-based")


@dataclass(frozen=True, slots=True)
class SignDescription:
    ordinal: int
    text: BilingualText
    topics_en: tuple[str, ...]
    topics_ru: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class DescribedPhoto:
    photo: BilingualText
    signs: tuple[SignDescription, ...]

    def __post_init__(self) -> None:
        expected = tuple(range(len(self.signs)))
        actual = tuple(sign.ordinal for sign in self.signs)
        if actual != expected:
            raise ValueError("Sign descriptions must be contiguous and zero-based")


@dataclass(frozen=True, slots=True)
class ProviderIdentity:
    provider: str
    model: str
    namespace: str = "default"

    def __post_init__(self) -> None:
        if not self.provider.strip() or not self.model.strip() or not self.namespace.strip():
            raise ValueError("Provider identity fields cannot be empty")


@dataclass(frozen=True, slots=True)
class Embedding:
    identity: ProviderIdentity
    values: tuple[float, ...]

    def __post_init__(self) -> None:
        if not self.values:
            raise ValueError("Embedding cannot be empty")
        if not all(isfinite(value) for value in self.values):
            raise ValueError("Embedding values must be finite")


@dataclass(frozen=True, slots=True)
class StoredAsset:
    sha256: str
    relative_path: str
    media_type: str
    byte_size: int
    width: int | None
    height: int | None
    original_filename: str | None


@dataclass(frozen=True, slots=True)
class PreparedImage:
    data: bytes
    media_type: str
    width: int
    height: int
    source_width: int
    source_height: int


@dataclass(frozen=True, slots=True)
class ImageMetadata:
    raw_json: str
    captured_at: str | None = None
    latitude: float | None = None
    longitude: float | None = None
    altitude_m: float | None = None
    camera_make: str | None = None
    camera_model: str | None = None
    lens_model: str | None = None
    software: str | None = None
    orientation: int | None = None
    extractor: str = "unknown"


@dataclass(frozen=True, slots=True)
class TelegramFile:
    file_id: str | None = None
    file_unique_id: str | None = None
    file_name: str | None = None
    mime_type: str | None = None
    file_size: int | None = None


@dataclass(frozen=True, slots=True)
class TelegramMessageRef:
    role: TelegramMessageRole
    chat_id: int
    message_id: int
    sent_at: str | None = None
    edited_at: str | None = None
    media_group_id: str | None = None
    caption: str | None = None
    file: TelegramFile | None = None


@dataclass(frozen=True, slots=True)
class IngestRequest:
    data: bytes
    filename: str | None
    declared_media_type: str | None
    source_kind: SourceKind
    source_key: str
    source_message: TelegramMessageRef | None = None
    assume_sign: bool = False
    initial_sign_status: ReviewStatus = ReviewStatus.PENDING
    existing_channel_message: TelegramMessageRef | None = None


@dataclass(frozen=True, slots=True)
class PreparedSign:
    item_id: str
    asset: StoredAsset
    observation: SignObservation
    crop_box: BoundingBox
    description: SignDescription
    embedding: Embedding
    status: ReviewStatus
    published_message: TelegramMessageRef | None = None


@dataclass(frozen=True, slots=True)
class PreparedIngestion:
    photo_item_id: str
    source_kind: SourceKind
    source_key: str
    asset: StoredAsset
    metadata: ImageMetadata
    observation: PhotoObservation
    description: BilingualText
    embedding: Embedding
    vision_identity: ProviderIdentity
    text_identity: ProviderIdentity
    signs: tuple[PreparedSign, ...]
    source_message: TelegramMessageRef | None = None


@dataclass(frozen=True, slots=True)
class IngestedSign:
    item_id: str
    ordinal: int
    status: ReviewStatus
    crop_relative_path: str
    description: BilingualText


@dataclass(frozen=True, slots=True)
class IngestionResult:
    photo_item_id: str
    original_relative_path: str
    description: BilingualText
    metadata: ImageMetadata
    signs: tuple[IngestedSign, ...]
    already_existed: bool = False


@dataclass(frozen=True, slots=True)
class ReviewCandidate:
    item_id: str
    review_token: str
    ordinal: int
    status: ReviewStatus
    crop_relative_path: str
    photo_description: BilingualText
    sign_description: BilingualText
    topics_en: tuple[str, ...]
    topics_ru: tuple[str, ...]
    visible_text: tuple[str, ...]
    metadata: ImageMetadata
    last_error: str | None = None


@dataclass(frozen=True, slots=True)
class PublishedMessage:
    chat_id: int
    message_id: int
    sent_at: str | None
    file: TelegramFile | None = None


@dataclass(frozen=True, slots=True)
class SearchDocument:
    item_id: str
    kind: ItemKind
    text: BilingualText
    asset_relative_path: str
    embedding: Embedding
    status: ReviewStatus


@dataclass(frozen=True, slots=True)
class SearchHit:
    item_id: str
    kind: ItemKind
    description: BilingualText
    asset_relative_path: str
    status: ReviewStatus
    score: float


@dataclass(frozen=True, slots=True)
class ItemEmbedding:
    item_id: str
    embedding: Embedding


@dataclass(frozen=True, slots=True)
class SignEmbeddingDocument:
    item_id: str
    parent_photo_id: str
    text: BilingualText
    topics_en: tuple[str, ...]
    topics_ru: tuple[str, ...]
    asset_relative_path: str
    status: ReviewStatus
    semantic_embedding: Embedding
    visual_embedding: Embedding | None
