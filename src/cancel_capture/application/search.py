from __future__ import annotations

import math

from cancel_capture.models import ItemKind, SearchHit
from cancel_capture.ports import AssetStore, CatalogRepository, EmbeddingProvider


class SearchService:
    def __init__(
        self,
        catalog: CatalogRepository,
        embeddings: EmbeddingProvider,
        assets: AssetStore,
    ) -> None:
        self._catalog = catalog
        self._embeddings = embeddings
        self._assets = assets

    async def search(
        self, query: str, *, kind: ItemKind | None = None, limit: int = 20
    ) -> tuple[SearchHit, ...]:
        clean_query = query.strip()
        if not clean_query or limit <= 0:
            return ()
        query_vectors = await self._embeddings.embed((clean_query,))
        query_vector = query_vectors[0]
        documents = self._catalog.list_search_documents(
            query_vector.identity,
            len(query_vector.values),
            kind.value if kind is not None else None,
        )
        lexical = self._catalog.lexical_matches(
            clean_query,
            kind.value if kind is not None else None,
        )
        scored: list[SearchHit] = []
        for document in documents:
            semantic = self._cosine(query_vector.values, document.embedding.values)
            lexical_bonus = 0.05 if document.item_id in lexical else 0.0
            scored.append(
                SearchHit(
                    item_id=document.item_id,
                    kind=document.kind,
                    description=document.text,
                    asset_relative_path=document.asset_relative_path,
                    status=document.status,
                    score=semantic + lexical_bonus,
                )
            )
        scored.sort(key=lambda hit: (-hit.score, hit.item_id))
        return tuple(scored[:limit])

    @staticmethod
    def _cosine(left: tuple[float, ...], right: tuple[float, ...]) -> float:
        if len(left) != len(right):
            raise ValueError("Cannot compare embeddings with different dimensions")
        left_norm = math.sqrt(sum(value * value for value in left))
        right_norm = math.sqrt(sum(value * value for value in right))
        if left_norm == 0.0 or right_norm == 0.0:
            return 0.0
        return sum(a * b for a, b in zip(left, right, strict=True)) / (left_norm * right_norm)
