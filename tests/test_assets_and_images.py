from io import BytesIO

import numpy as np
import pytest
from PIL import Image

from cancel_capture.adapters.filesystem import ContentAddressedAssetStore
from cancel_capture.adapters.image import PillowImageProcessor
from cancel_capture.adapters.telegram_photo import (
    TELEGRAM_PHOTO_MAX_ASPECT_RATIO,
    TELEGRAM_PHOTO_MAX_BYTES,
    TELEGRAM_PHOTO_MAX_DIMENSION_SUM,
    render_telegram_photo,
)
from cancel_capture.errors import ImageTooLargeError, UnsupportedImageError
from cancel_capture.models import BoundingBox
from tests.fakes import sample_jpeg


def test_original_is_byte_exact_and_filename_cannot_escape_volume(tmp_path) -> None:
    data = sample_jpeg()
    store = ContentAddressedAssetStore(tmp_path / "private", max_upload_bytes=len(data) + 1)
    asset = store.save_original(data, "../../camera/original.jpg", "image/jpeg")

    assert store.resolve(asset.relative_path).read_bytes() == data
    assert asset.original_filename == "original.jpg"
    assert asset.relative_path.startswith("assets/originals/")
    with pytest.raises(ValueError):
        store.resolve("../../outside.jpg")


def test_asset_limit_is_checked_before_write(tmp_path) -> None:
    store = ContentAddressedAssetStore(tmp_path / "private", max_upload_bytes=10)
    with pytest.raises(ImageTooLargeError):
        store.save_original(b"x" * 11, "large.jpg", "image/jpeg")


def test_analysis_and_crop_outputs_have_no_exif(tmp_path) -> None:
    image = Image.new("RGB", (300, 200), "white")
    exif = Image.Exif()
    exif[271] = "Secret Camera"
    source = tmp_path / "with-exif.jpg"
    image.save(source, exif=exif)

    processor = PillowImageProcessor(max_image_pixels=1_000_000)
    prepared = processor.prepare(source)
    with Image.open(BytesIO(prepared.data)) as analyzed:
        assert len(analyzed.getexif()) == 0

    crop_data, crop_box, width, height = processor.crop(
        source, prepared, BoundingBox(0.1, 0.1, 0.9, 0.9)
    )
    assert crop_box.left <= 0.1
    assert width > 0 and height > 0
    with Image.open(BytesIO(crop_data)) as crop:
        assert crop.format == "JPEG"
        assert len(crop.getexif()) == 0


def test_crop_uses_oriented_full_resolution_source(tmp_path) -> None:
    image = Image.new("RGB", (400, 200), "white")
    exif = Image.Exif()
    exif[271] = "Secret Camera"
    exif[274] = 6
    source = tmp_path / "rotated.jpg"
    image.save(source, exif=exif)

    processor = PillowImageProcessor(max_image_pixels=1_000_000, max_analysis_side=100)
    prepared = processor.prepare(source)

    assert (prepared.width, prepared.height) == (50, 100)
    assert (prepared.source_width, prepared.source_height) == (200, 400)

    crop_data, _, width, height = processor.crop(source, prepared, BoundingBox.full_frame())

    assert (width, height) == (200, 400)
    with Image.open(BytesIO(crop_data)) as crop:
        assert crop.size == (200, 400)
        assert len(crop.getexif()) == 0


def test_telegram_rendition_is_legal_deterministic_and_leaves_archive_unchanged(
    tmp_path,
) -> None:
    image = Image.new("RGB", (12_000, 100), "red")
    exif = Image.Exif()
    exif[271] = "Private Camera"
    archive = tmp_path / "archival-crop.jpg"
    image.save(archive, format="JPEG", quality=95, exif=exif)
    archival_bytes = archive.read_bytes()

    first = render_telegram_photo(archive)
    second = render_telegram_photo(archive)

    assert first == second
    assert archive.read_bytes() == archival_bytes
    assert len(first) < TELEGRAM_PHOTO_MAX_BYTES
    with Image.open(BytesIO(first)) as rendition:
        assert rendition.format == "JPEG"
        assert rendition.width + rendition.height <= TELEGRAM_PHOTO_MAX_DIMENSION_SUM
        assert max(rendition.size) / min(rendition.size) <= TELEGRAM_PHOTO_MAX_ASPECT_RATIO
        assert len(rendition.getexif()) == 0


def test_telegram_rendition_recompresses_noisy_crop_below_size_limit(tmp_path) -> None:
    pixels = np.random.default_rng(2026).integers(0, 256, size=(4_000, 4_000, 3), dtype=np.uint8)
    archive = tmp_path / "large-noisy-crop.jpg"
    Image.fromarray(pixels).save(archive, format="JPEG", quality=100)

    assert archive.stat().st_size > TELEGRAM_PHOTO_MAX_BYTES

    rendition = render_telegram_photo(archive)

    assert len(rendition) < TELEGRAM_PHOTO_MAX_BYTES
    with Image.open(BytesIO(rendition)) as opened:
        assert opened.width + opened.height <= TELEGRAM_PHOTO_MAX_DIMENSION_SUM
        assert max(opened.size) / min(opened.size) <= TELEGRAM_PHOTO_MAX_ASPECT_RATIO
        assert len(opened.getexif()) == 0


def test_non_image_is_rejected(tmp_path) -> None:
    source = tmp_path / "not-an-image.bin"
    source.write_bytes(b"not an image")
    with pytest.raises(UnsupportedImageError):
        PillowImageProcessor(max_image_pixels=1_000_000).prepare(source)
