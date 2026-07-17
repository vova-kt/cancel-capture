from __future__ import annotations

import math
import warnings
from io import BytesIO
from pathlib import Path

from PIL import Image, ImageOps, UnidentifiedImageError

from cancel_capture.errors import ImageTooLargeError, UnsupportedImageError

TELEGRAM_PHOTO_MAX_BYTES = 10_000_000
TELEGRAM_PHOTO_MAX_DIMENSION_SUM = 10_000
TELEGRAM_PHOTO_MAX_ASPECT_RATIO = 20

_JPEG_QUALITIES = (90, 80, 70, 60, 50, 40, 30)
_SIZE_REDUCTION_FACTOR = 0.8


def render_telegram_photo(path: Path) -> bytes:
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", Image.DecompressionBombWarning)
            with Image.open(path) as opened:
                image = ImageOps.exif_transpose(opened).convert("RGB")
    except (UnidentifiedImageError, OSError) as error:
        raise UnsupportedImageError("The archival crop is not a supported image") from error
    except Image.DecompressionBombWarning as error:
        raise ImageTooLargeError("The archival crop exceeds Pillow's safe pixel limit") from error

    image = _pad_aspect_ratio(image)
    image = _fit_dimension_sum(image)
    while True:
        for quality in _JPEG_QUALITIES:
            rendition = _encode_jpeg(image, quality)
            if len(rendition) < TELEGRAM_PHOTO_MAX_BYTES:
                return rendition
        image = _shrink(image)


def _pad_aspect_ratio(image: Image.Image) -> Image.Image:
    width, height = image.size
    if width > height * TELEGRAM_PHOTO_MAX_ASPECT_RATIO:
        canvas = Image.new(
            "RGB",
            (width, math.ceil(width / TELEGRAM_PHOTO_MAX_ASPECT_RATIO)),
            "white",
        )
        canvas.paste(image, (0, (canvas.height - height) // 2))
        return canvas
    if height > width * TELEGRAM_PHOTO_MAX_ASPECT_RATIO:
        canvas = Image.new(
            "RGB",
            (math.ceil(height / TELEGRAM_PHOTO_MAX_ASPECT_RATIO), height),
            "white",
        )
        canvas.paste(image, ((canvas.width - width) // 2, 0))
        return canvas
    return image


def _fit_dimension_sum(image: Image.Image) -> Image.Image:
    while image.width + image.height > TELEGRAM_PHOTO_MAX_DIMENSION_SUM:
        scale = TELEGRAM_PHOTO_MAX_DIMENSION_SUM / (image.width + image.height)
        resized = image.resize(
            (
                max(1, math.floor(image.width * scale)),
                max(1, math.floor(image.height * scale)),
            ),
            Image.Resampling.LANCZOS,
        )
        image = _pad_aspect_ratio(resized)
    return image


def _shrink(image: Image.Image) -> Image.Image:
    if image.size == (1, 1):
        raise ImageTooLargeError("Telegram photo rendition cannot fit below 10 MB")
    resized = image.resize(
        (
            max(1, math.floor(image.width * _SIZE_REDUCTION_FACTOR)),
            max(1, math.floor(image.height * _SIZE_REDUCTION_FACTOR)),
        ),
        Image.Resampling.LANCZOS,
    )
    return _pad_aspect_ratio(resized)


def _encode_jpeg(image: Image.Image, quality: int) -> bytes:
    output = BytesIO()
    image.save(output, format="JPEG", quality=quality, optimize=True)
    return output.getvalue()
