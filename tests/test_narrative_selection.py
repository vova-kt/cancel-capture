from __future__ import annotations

import pytest

from cancel_capture.application.narrative_selection import (
    NarrativeSelectionService,
    SimilarityMode,
)
from cancel_capture.models import (
    BilingualText,
    Embedding,
    ItemEmbedding,
    ProviderIdentity,
    ReviewStatus,
    SignEmbeddingDocument,
)

SEMANTIC_IDENTITY = ProviderIdentity("test", "semantic-v1")
VISUAL_IDENTITY = ProviderIdentity("test", "visual-v1")


def _document(
    *,
    item_id: str,
    parent: str,
    semantic: tuple[float, ...],
    visual: tuple[float, ...] | None = (1.0, 0.0),
    status: ReviewStatus = ReviewStatus.PUBLISHED,
    topics: tuple[str, ...] = ("prohibition",),
) -> SignEmbeddingDocument:
    return SignEmbeddingDocument(
        item_id=item_id,
        parent_photo_id=parent,
        text=BilingualText(en=item_id, ru=item_id),
        topics_en=topics,
        topics_ru=topics,
        asset_relative_path=f"assets/crops/{item_id}.jpg",
        status=status,
        semantic_embedding=Embedding(identity=SEMANTIC_IDENTITY, values=semantic),
        visual_embedding=(
            Embedding(identity=VISUAL_IDENTITY, values=visual) if visual is not None else None
        ),
    )


class InMemoryCatalog:
    def __init__(self, documents: tuple[SignEmbeddingDocument, ...]) -> None:
        self._documents = documents

    def list_sign_embedding_documents(self) -> tuple[SignEmbeddingDocument, ...]:
        return self._documents

    def upsert_visual_embeddings(  # pragma: no cover
        self, embeddings: tuple[ItemEmbedding, ...]
    ) -> None:
        raise NotImplementedError


def _service(documents: tuple[SignEmbeddingDocument, ...]) -> NarrativeSelectionService:
    return NarrativeSelectionService(InMemoryCatalog(documents))  # type: ignore[arg-type]


def test_select_excludes_similar_siblings_and_reports_eligible_count() -> None:
    anchor = _document(item_id="anchor", parent="photo-a", semantic=(1.0, 0.0))
    sibling = _document(item_id="sibling", parent="photo-a", semantic=(0.0, 1.0))
    close = _document(item_id="close", parent="photo-b", semantic=(0.99, 0.1))
    distant = _document(item_id="distant", parent="photo-c", semantic=(0.0, 1.0))
    farther = _document(item_id="farther", parent="photo-d", semantic=(-1.0, 0.0))

    service = _service((anchor, sibling, close, distant, farther))
    selection = service.select(
        "anchor",
        count=2,
        maximum_similarity=0.5,
        statuses=frozenset({ReviewStatus.PUBLISHED}),
        mode=SimilarityMode.SEMANTIC,
        seed=17,
    )

    picked = {companion.document.item_id for companion in selection.companions}
    assert picked.issubset({"distant", "farther"})
    assert selection.eligible_count == 2
    assert selection.requested_count == 2
    assert selection.anchor.is_anchor is True
    assert all(companion.similarity_to_anchor <= 0.5 for companion in selection.companions)


def test_select_is_deterministic_for_a_seed_and_uniform_over_reseeds() -> None:
    documents = tuple(
        _document(item_id=f"sign-{index}", parent=f"photo-{index}", semantic=(0.1, 0.9))
        for index in range(6)
    )
    anchor = _document(item_id="anchor", parent="photo-anchor", semantic=(1.0, 0.0))

    service = _service((anchor, *documents))

    first = service.select(
        "anchor",
        count=3,
        maximum_similarity=1.0,
        statuses=frozenset({ReviewStatus.PUBLISHED}),
        mode=SimilarityMode.SEMANTIC,
        seed=42,
    )
    repeated = service.select(
        "anchor",
        count=3,
        maximum_similarity=1.0,
        statuses=frozenset({ReviewStatus.PUBLISHED}),
        mode=SimilarityMode.SEMANTIC,
        seed=42,
    )
    other = service.select(
        "anchor",
        count=3,
        maximum_similarity=1.0,
        statuses=frozenset({ReviewStatus.PUBLISHED}),
        mode=SimilarityMode.SEMANTIC,
        seed=43,
    )

    def ids(selection):  # pyright: ignore [reportMissingParameterType, reportUnknownParameterType]
        return tuple(companion.document.item_id for companion in selection.companions)

    assert ids(first) == ids(repeated)
    assert ids(first) != ids(other)


def test_hybrid_similarity_falls_back_and_visual_mode_requires_vectors() -> None:
    anchor = _document(item_id="anchor", parent="a", semantic=(1.0, 0.0), visual=(1.0, 0.0))
    with_visual = _document(item_id="visual", parent="b", semantic=(0.1, 0.9), visual=(0.1, 0.9))
    without_visual = _document(item_id="text", parent="c", semantic=(0.0, 1.0), visual=None)

    service = _service((anchor, with_visual, without_visual))

    hybrid = service.select(
        "anchor",
        count=5,
        maximum_similarity=1.0,
        statuses=frozenset({ReviewStatus.PUBLISHED}),
        mode=SimilarityMode.HYBRID,
        seed=7,
    )
    hybrid_ids = {companion.document.item_id for companion in hybrid.companions}
    assert hybrid_ids == {"visual"}

    visual_only = service.select(
        "anchor",
        count=5,
        maximum_similarity=1.0,
        statuses=frozenset({ReviewStatus.PUBLISHED}),
        mode=SimilarityMode.VISUAL,
        seed=7,
    )
    assert {companion.document.item_id for companion in visual_only.companions} == {"visual"}


def test_random_anchor_returns_none_when_no_candidates_and_respects_seed() -> None:
    service = _service(())
    assert service.random_anchor(frozenset({ReviewStatus.PUBLISHED}), seed=1) is None

    documents = tuple(
        _document(item_id=f"sign-{index}", parent=f"photo-{index}", semantic=(1.0, 0.0))
        for index in range(4)
    )
    populated = _service(documents)
    first = populated.random_anchor(frozenset({ReviewStatus.PUBLISHED}), seed=99)
    repeated = populated.random_anchor(frozenset({ReviewStatus.PUBLISHED}), seed=99)
    assert first is not None and repeated is not None
    assert first.item_id == repeated.item_id


def test_list_eligible_similarities_returns_sorted_values_and_rejects_unknown_anchor() -> None:
    anchor = _document(item_id="anchor", parent="a", semantic=(1.0, 0.0))
    sibling = _document(item_id="sibling", parent="a", semantic=(0.0, 1.0))
    close = _document(item_id="close", parent="b", semantic=(0.9, 0.44))
    distant = _document(item_id="distant", parent="c", semantic=(0.0, 1.0))
    hidden = _document(
        item_id="hidden", parent="d", semantic=(-1.0, 0.0), status=ReviewStatus.REJECTED
    )

    service = _service((anchor, sibling, close, distant, hidden))
    similarities = service.list_eligible_similarities(
        "anchor",
        statuses=frozenset({ReviewStatus.PUBLISHED}),
        mode=SimilarityMode.SEMANTIC,
    )

    assert list(similarities) == sorted(similarities)
    assert len(similarities) == 2  # sibling and hidden excluded, close + distant included
    assert similarities[-1] > similarities[0]

    with pytest.raises(ValueError, match="Unknown narrative anchor"):
        service.list_eligible_similarities(
            "missing",
            statuses=frozenset({ReviewStatus.PUBLISHED}),
        )


def test_select_validates_inputs() -> None:
    anchor = _document(item_id="anchor", parent="a", semantic=(1.0, 0.0))
    service = _service((anchor,))
    with pytest.raises(ValueError, match="Unknown narrative anchor"):
        service.select(
            "missing",
            count=1,
            maximum_similarity=1.0,
            statuses=frozenset({ReviewStatus.PUBLISHED}),
            seed=0,
        )
    with pytest.raises(ValueError, match="between -1 and 1"):
        service.select(
            "anchor",
            count=1,
            maximum_similarity=2.0,
            statuses=frozenset({ReviewStatus.PUBLISHED}),
            seed=0,
        )
    with pytest.raises(ValueError, match="cannot be negative"):
        service.select(
            "anchor",
            count=-1,
            maximum_similarity=1.0,
            statuses=frozenset({ReviewStatus.PUBLISHED}),
            seed=0,
        )
