from __future__ import annotations

import math
import random
from dataclasses import dataclass
from enum import StrEnum

from cancel_capture.models import Embedding, ReviewStatus, SignEmbeddingDocument
from cancel_capture.ports import CatalogRepository


class SimilarityMode(StrEnum):
    HYBRID = "hybrid"
    SEMANTIC = "semantic"
    VISUAL = "visual"


@dataclass(frozen=True, slots=True)
class SelectedNarrativeSign:
    document: SignEmbeddingDocument
    similarity_to_anchor: float
    is_anchor: bool = False


@dataclass(frozen=True, slots=True)
class NarrativeSelection:
    anchor: SelectedNarrativeSign
    companions: tuple[SelectedNarrativeSign, ...]
    eligible_count: int
    requested_count: int


class NarrativeSelectionService:
    def __init__(self, catalog: CatalogRepository) -> None:
        self._catalog = catalog

    def list_anchors(
        self, statuses: frozenset[ReviewStatus]
    ) -> tuple[SignEmbeddingDocument, ...]:
        return tuple(
            document
            for document in self._catalog.list_sign_embedding_documents()
            if document.status in statuses
        )

    def random_anchor(
        self,
        statuses: frozenset[ReviewStatus],
        *,
        seed: int,
    ) -> SignEmbeddingDocument | None:
        candidates = self.list_anchors(statuses)
        if not candidates:
            return None
        return random.Random(seed).choice(candidates)

    def select(
        self,
        anchor_id: str,
        *,
        count: int,
        maximum_similarity: float,
        statuses: frozenset[ReviewStatus],
        mode: SimilarityMode = SimilarityMode.HYBRID,
        semantic_weight: float = 0.65,
        seed: int,
    ) -> NarrativeSelection:
        if count < 0:
            raise ValueError("Narrative companion count cannot be negative")
        if not -1.0 <= maximum_similarity <= 1.0:
            raise ValueError("Maximum similarity must be between -1 and 1")
        if not 0.0 <= semantic_weight <= 1.0:
            raise ValueError("Semantic weight must be between 0 and 1")

        documents = self._catalog.list_sign_embedding_documents()
        anchor = next((document for document in documents if document.item_id == anchor_id), None)
        if anchor is None:
            raise ValueError(f"Unknown narrative anchor: {anchor_id}")

        eligible: list[SelectedNarrativeSign] = []
        for candidate in documents:
            if candidate.item_id == anchor.item_id:
                continue
            if candidate.parent_photo_id == anchor.parent_photo_id:
                continue
            if candidate.status not in statuses:
                continue
            similarity = self._similarity(anchor, candidate, mode, semantic_weight)
            if similarity is None or similarity > maximum_similarity:
                continue
            eligible.append(
                SelectedNarrativeSign(
                    document=candidate,
                    similarity_to_anchor=similarity,
                )
            )

        eligible.sort(key=lambda selected: selected.document.item_id)
        picked = random.Random(seed).sample(eligible, k=min(count, len(eligible)))
        return NarrativeSelection(
            anchor=SelectedNarrativeSign(anchor, similarity_to_anchor=1.0, is_anchor=True),
            companions=tuple(picked),
            eligible_count=len(eligible),
            requested_count=count,
        )

    @classmethod
    def _similarity(
        cls,
        anchor: SignEmbeddingDocument,
        candidate: SignEmbeddingDocument,
        mode: SimilarityMode,
        semantic_weight: float,
    ) -> float | None:
        semantic = cls._compatible_cosine(
            anchor.semantic_embedding, candidate.semantic_embedding
        )
        visual: float | None = None
        if anchor.visual_embedding is not None and candidate.visual_embedding is not None:
            visual = cls._compatible_cosine(anchor.visual_embedding, candidate.visual_embedding)
        if mode is SimilarityMode.SEMANTIC:
            return semantic
        if mode is SimilarityMode.VISUAL:
            return visual
        if semantic is None or visual is None:
            return None
        return semantic_weight * semantic + (1.0 - semantic_weight) * visual

    @staticmethod
    def _compatible_cosine(left: Embedding, right: Embedding) -> float | None:
        if left.identity != right.identity or len(left.values) != len(right.values):
            return None
        left_norm = math.sqrt(sum(value * value for value in left.values))
        right_norm = math.sqrt(sum(value * value for value in right.values))
        if left_norm == 0.0 or right_norm == 0.0:
            return 0.0
        return sum(a * b for a, b in zip(left.values, right.values, strict=True)) / (
            left_norm * right_norm
        )
