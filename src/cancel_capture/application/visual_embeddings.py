from __future__ import annotations

from cancel_capture.errors import ProviderResponseError
from cancel_capture.models import ItemEmbedding, ProviderIdentity, SignEmbeddingDocument
from cancel_capture.ports import (
    AssetStore,
    CatalogRepository,
    ImageProcessor,
    VisualEmbeddingProvider,
)


class VisualEmbeddingService:
    def __init__(
        self,
        catalog: CatalogRepository,
        assets: AssetStore,
        images: ImageProcessor,
        provider: VisualEmbeddingProvider,
        *,
        batch_size: int = 64,
    ) -> None:
        if batch_size <= 0:
            raise ValueError("Visual embedding batch size must be positive")
        self._catalog = catalog
        self._assets = assets
        self._images = images
        self._provider = provider
        self._batch_size = batch_size

    @property
    def identity(self) -> ProviderIdentity:
        return self._provider.identity

    @property
    def dimensions(self) -> int:
        return self._provider.dimensions

    def ensure_current(self) -> int:
        pending = tuple(
            document
            for document in self._catalog.list_sign_embedding_documents()
            if not self._is_current(document)
        )
        updated = 0
        for offset in range(0, len(pending), self._batch_size):
            batch = pending[offset : offset + self._batch_size]
            prepared = tuple(
                self._images.prepare(self._assets.resolve(document.asset_relative_path))
                for document in batch
            )
            vectors = self._provider.embed(prepared)
            if len(vectors) != len(batch):
                raise ProviderResponseError(
                    "Visual embedding count does not match the requested sign crops"
                )
            self._catalog.upsert_visual_embeddings(
                tuple(
                    ItemEmbedding(document.item_id, vector)
                    for document, vector in zip(batch, vectors, strict=True)
                )
            )
            updated += len(batch)
        return updated

    def _is_current(self, document: SignEmbeddingDocument) -> bool:
        embedding = document.visual_embedding
        return (
            embedding is not None
            and embedding.identity == self.identity
            and len(embedding.values) == self.dimensions
        )
