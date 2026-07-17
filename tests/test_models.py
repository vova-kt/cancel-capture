import pytest

from cancel_capture.models import BilingualText, BoundingBox


def test_bounding_box_rejects_invalid_coordinates() -> None:
    with pytest.raises(ValueError):
        BoundingBox(0.8, 0.1, 0.2, 0.9)
    with pytest.raises(ValueError):
        BoundingBox(-0.1, 0.1, 0.2, 0.9)


def test_bounding_box_expansion_clamps_to_frame() -> None:
    box = BoundingBox(0.02, 0.03, 0.5, 0.6).expanded(0.2)
    assert box.left == 0.0
    assert box.top == 0.0
    assert box.right > 0.5
    assert box.bottom > 0.6


def test_bilingual_search_text_keeps_languages_explicit() -> None:
    text = BilingualText(en="No bicycles", ru="Велосипеды запрещены")
    assert text.search_text() == ("English:\nNo bicycles\n\nRussian:\nВелосипеды запрещены")
