from __future__ import annotations

from pathlib import Path
from typing import Protocol

from cancel_capture.models import (
    BoundingBox,
    DescribedPhoto,
    Embedding,
    ImageMetadata,
    IngestionResult,
    ItemEmbedding,
    PhotoObservation,
    PreparedImage,
    PreparedIngestion,
    ProviderIdentity,
    PublishedMessage,
    ReviewCandidate,
    ReviewStatus,
    SearchDocument,
    SignEmbeddingDocument,
    StoredAsset,
    TelegramMessageRef,
)


class VisionProvider(Protocol):
    @property
    def identity(self) -> ProviderIdentity: ...

    async def inspect(self, image: PreparedImage) -> PhotoObservation: ...


class TextProvider(Protocol):
    @property
    def identity(self) -> ProviderIdentity: ...

    async def describe(self, observation: PhotoObservation) -> DescribedPhoto: ...


class EmbeddingProvider(Protocol):
    @property
    def identity(self) -> ProviderIdentity: ...

    async def embed(self, texts: tuple[str, ...]) -> tuple[Embedding, ...]: ...


class VisualEmbeddingProvider(Protocol):
    @property
    def identity(self) -> ProviderIdentity: ...

    @property
    def dimensions(self) -> int: ...

    def embed(self, images: tuple[PreparedImage, ...]) -> tuple[Embedding, ...]: ...


class AssetStore(Protocol):
    def save_original(
        self, data: bytes, filename: str | None, declared_media_type: str | None
    ) -> StoredAsset: ...

    def save_crop(self, data: bytes, width: int, height: int) -> StoredAsset: ...

    def resolve(self, relative_path: str) -> Path: ...


class ImageProcessor(Protocol):
    def prepare(self, path: Path) -> PreparedImage: ...

    def crop(
        self, source_path: Path, analysis_image: PreparedImage, box: BoundingBox
    ) -> tuple[bytes, BoundingBox, int, int]: ...


class MetadataExtractor(Protocol):
    def extract(self, path: Path) -> ImageMetadata: ...


class ChannelPublisher(Protocol):
    async def publish(self, candidate: ReviewCandidate) -> PublishedMessage: ...


class CatalogRepository(Protocol):
    def initialize(self) -> None: ...

    def find_ingestion(self, source_key: str) -> IngestionResult | None: ...

    def insert_ingestion(self, ingestion: PreparedIngestion) -> IngestionResult: ...

    def get_candidate(self, item_id: str) -> ReviewCandidate: ...

    def record_preview(self, item_id: str, message: TelegramMessageRef) -> None: ...

    def has_preview(self, item_id: str) -> bool: ...

    def claim_publish(
        self,
        item_id: str,
        review_token: str,
        expected_status: ReviewStatus,
        actor_user_id: int,
    ) -> ReviewCandidate: ...

    def recover_interrupted_publishes(self) -> int: ...

    def complete_publish(
        self, item_id: str, actor_user_id: int, message: PublishedMessage
    ) -> None: ...

    def reconcile_publish(
        self, item_id: str, actor_user_id: int, message: PublishedMessage
    ) -> None: ...

    def fail_publish(self, item_id: str, actor_user_id: int, error: str) -> None: ...

    def reject(self, item_id: str, review_token: str, actor_user_id: int) -> ReviewCandidate: ...

    def list_search_documents(
        self, identity: ProviderIdentity, dimensions: int, kind: str | None = None
    ) -> tuple[SearchDocument, ...]: ...

    def lexical_matches(self, query: str, kind: str | None = None) -> frozenset[str]: ...

    def list_candidates(
        self, status: ReviewStatus | None = None
    ) -> tuple[ReviewCandidate, ...]: ...

    def list_sign_embedding_documents(self) -> tuple[SignEmbeddingDocument, ...]: ...

    def upsert_visual_embeddings(self, embeddings: tuple[ItemEmbedding, ...]) -> None: ...
