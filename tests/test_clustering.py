import math
from io import BytesIO

import pytest
from PIL import Image, ImageDraw

from cancel_capture.adapters.visual_embedding import PillowVisualEmbeddingProvider
from cancel_capture.application.clustering import average_linkage_cosine
from cancel_capture.errors import ImageTooLargeError, UnsupportedImageError
from cancel_capture.models import (
    BilingualText,
    Embedding,
    ItemKind,
    PreparedImage,
    ProviderIdentity,
    ReviewStatus,
    SearchDocument,
)

IDENTITY = ProviderIdentity("test", "visual-v1")


def _document(item_id: str, values: tuple[float, ...]) -> SearchDocument:
    return SearchDocument(
        item_id=item_id,
        kind=ItemKind.SIGN,
        text=BilingualText(en=item_id, ru=item_id),
        asset_relative_path=f"assets/crops/{item_id}.jpg",
        embedding=Embedding(identity=IDENTITY, values=values),
        status=ReviewStatus.PUBLISHED,
    )


def _prepared(image: Image.Image) -> PreparedImage:
    output = BytesIO()
    image.save(output, format="PNG")
    return PreparedImage(
        data=output.getvalue(),
        media_type="image/png",
        width=image.width,
        height=image.height,
        source_width=image.width,
        source_height=image.height,
    )


def test_average_linkage_cut_groups_nearby_vectors() -> None:
    documents = (
        _document("b", (0.99, 0.1)),
        _document("d", (0.1, 0.99)),
        _document("a", (1.0, 0.0)),
        _document("c", (0.0, 1.0)),
    )

    clustering = average_linkage_cosine(documents)

    assert tuple(group.item_ids for group in clustering.cut(2)) == (("a", "b"), ("c", "d"))
    assert [merge.distance for merge in clustering.merges] == sorted(
        merge.distance for merge in clustering.merges
    )
    assert clustering.cut(1)[0].item_ids == ("a", "b", "c", "d")
    assert tuple(group.item_ids for group in clustering.cut(4)) == (
        ("a",),
        ("b",),
        ("c",),
        ("d",),
    )


def test_clustering_is_deterministic_for_tied_distances_and_input_order() -> None:
    documents = (
        _document("d", (-1.0, 0.0)),
        _document("c", (0.0, -1.0)),
        _document("b", (0.0, 1.0)),
        _document("a", (1.0, 0.0)),
    )

    forward = average_linkage_cosine(documents)
    reverse = average_linkage_cosine(tuple(reversed(documents)))

    assert forward == reverse
    assert forward.merges[0].left_node_id == 0
    assert forward.merges[0].right_node_id == 1


def test_clustering_rejects_mixed_or_malformed_document_sets() -> None:
    different_identity = SearchDocument(
        item_id="b",
        kind=ItemKind.SIGN,
        text=BilingualText(en="b", ru="b"),
        asset_relative_path="b.jpg",
        embedding=Embedding(identity=ProviderIdentity("other", "visual-v1"), values=(1.0, 0.0)),
        status=ReviewStatus.PUBLISHED,
    )
    with pytest.raises(ValueError, match="provider identities"):
        average_linkage_cosine((_document("a", (1.0, 0.0)), different_identity))
    with pytest.raises(ValueError, match="dimensions"):
        average_linkage_cosine((_document("a", (1.0, 0.0)), _document("b", (1.0,))))
    with pytest.raises(ValueError, match="unique item IDs"):
        average_linkage_cosine((_document("a", (1.0, 0.0)), _document("a", (0.0, 1.0))))


def test_empty_singleton_and_zero_vector_cuts_are_well_defined() -> None:
    empty = average_linkage_cosine(())
    assert empty.cut(0) == ()
    assert empty.dendrogram().segments == ()
    with pytest.raises(ValueError, match="empty"):
        empty.cut(1)

    singleton = average_linkage_cosine((_document("only", (0.0, 0.0)),))
    assert singleton.cut(1)[0].item_ids == ("only",)
    with pytest.raises(ValueError, match="between"):
        singleton.cut(2)

    zeros = average_linkage_cosine((_document("a", (0.0, 0.0)), _document("b", (0.0, 0.0))))
    assert zeros.merges[0].distance == 1.0


def test_dendrogram_geometry_is_renderer_independent() -> None:
    clustering = average_linkage_cosine(
        (
            _document("a", (1.0, 0.0)),
            _document("b", (0.9, 0.1)),
            _document("c", (0.0, 1.0)),
        )
    )

    geometry = clustering.dendrogram()

    assert set(geometry.leaf_item_ids) == {"a", "b", "c"}
    assert len(geometry.segments) == 3 * (len(clustering.documents) - 1)
    assert geometry.max_distance == clustering.merges[-1].distance
    assert all(
        math.isfinite(coordinate)
        for segment in geometry.segments
        for coordinate in (segment.x_start, segment.y_start, segment.x_end, segment.y_end)
    )
    assert sum(segment.y_start == segment.y_end for segment in geometry.segments) == 2


def test_visual_embedding_is_deterministic_normalized_and_image_sensitive() -> None:
    red = Image.new("RGB", (120, 80), "white")
    red_draw = ImageDraw.Draw(red)
    red_draw.ellipse((30, 10, 90, 70), outline="red", width=8)
    red_draw.line((40, 60, 80, 20), fill="red", width=8)
    blue = Image.new("RGB", (120, 80), "white")
    blue_draw = ImageDraw.Draw(blue)
    blue_draw.rectangle((30, 10, 90, 70), outline="blue", width=8)
    provider = PillowVisualEmbeddingProvider(max_image_pixels=100_000)
    red_image = _prepared(red)
    blue_image = _prepared(blue)

    first, repeated, different = provider.embed((red_image, red_image, blue_image))

    assert first == repeated
    assert first.identity == provider.identity
    assert len(first.values) == provider.dimensions
    assert math.isclose(math.sqrt(sum(value * value for value in first.values)), 1.0)
    assert first.values != different.values


def test_visual_embedding_validates_configuration_and_images() -> None:
    with pytest.raises(ValueError, match="pixels"):
        PillowVisualEmbeddingProvider(max_image_pixels=0)
    with pytest.raises(ValueError, match="grid"):
        PillowVisualEmbeddingProvider(max_image_pixels=100, grid_size=3)

    provider = PillowVisualEmbeddingProvider(max_image_pixels=100)
    with pytest.raises(UnsupportedImageError):
        provider.embed_one(
            PreparedImage(
                data=b"not an image",
                media_type="application/octet-stream",
                width=1,
                height=1,
                source_width=1,
                source_height=1,
            )
        )

    with pytest.raises(ImageTooLargeError):
        provider.embed_one(_prepared(Image.new("RGB", (11, 10), "white")))
