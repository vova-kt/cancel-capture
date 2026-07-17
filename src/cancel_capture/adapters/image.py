# pyright: reportMissingTypeStubs=false

from __future__ import annotations

import warnings
from collections.abc import Callable
from io import BytesIO
from pathlib import Path
from typing import cast

import numpy as np
from numpy.typing import NDArray
from PIL import Image, ImageOps, UnidentifiedImageError
from pillow_heif import (
    register_heif_opener,  # pyright: ignore[reportMissingTypeStubs, reportUnknownVariableType]
)

from cancel_capture.errors import ImageTooLargeError, UnsupportedImageError
from cancel_capture.models import BoundingBox, PreparedImage

cast(Callable[[], None], register_heif_opener)()


class PillowImageProcessor:
    def __init__(self, max_image_pixels: int, max_analysis_side: int = 4096) -> None:
        self._max_image_pixels = max_image_pixels
        self._max_analysis_side = max_analysis_side

    def prepare(self, path: Path) -> PreparedImage:
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("error", Image.DecompressionBombWarning)
                with Image.open(path) as opened:
                    if opened.width * opened.height > self._max_image_pixels:
                        raise ImageTooLargeError(
                            f"Decoded image has {opened.width * opened.height} pixels; "
                            f"limit is {self._max_image_pixels}"
                        )
                    oriented = ImageOps.exif_transpose(opened)
                    source_width, source_height = oriented.size
                    rgb = oriented.convert("RGB")
                    rgb.thumbnail(
                        (self._max_analysis_side, self._max_analysis_side),
                        Image.Resampling.LANCZOS,
                    )
                    output = BytesIO()
                    rgb.save(output, format="JPEG", quality=92, optimize=True)
                    return PreparedImage(
                        data=output.getvalue(),
                        media_type="image/jpeg",
                        width=rgb.width,
                        height=rgb.height,
                        source_width=source_width,
                        source_height=source_height,
                    )
        except (UnidentifiedImageError, OSError) as error:
            raise UnsupportedImageError("The uploaded file is not a supported image") from error
        except Image.DecompressionBombWarning as error:
            raise ImageTooLargeError("The image exceeds Pillow's safe pixel limit") from error

    def crop(
        self, source_path: Path, analysis_image: PreparedImage, box: BoundingBox
    ) -> tuple[bytes, BoundingBox, int, int]:
        with Image.open(BytesIO(analysis_image.data)) as opened:
            analysis_rgb = opened.convert("RGB")
            expanded = box.expanded(0.12)
            refined = self._refine_red_box(np.asarray(analysis_rgb, dtype=np.uint8), expanded)

        try:
            with warnings.catch_warnings():
                warnings.simplefilter("error", Image.DecompressionBombWarning)
                with Image.open(source_path) as opened:
                    oriented = ImageOps.exif_transpose(opened)
                    rgb = oriented.convert("RGB")
        except (UnidentifiedImageError, OSError) as error:
            raise UnsupportedImageError("The uploaded file is not a supported image") from error
        except Image.DecompressionBombWarning as error:
            raise ImageTooLargeError("The image exceeds Pillow's safe pixel limit") from error

        if (rgb.width, rgb.height) != (
            analysis_image.source_width,
            analysis_image.source_height,
        ):
            raise UnsupportedImageError("The source image changed during analysis")

        left = max(0, min(rgb.width - 1, round(refined.left * rgb.width)))
        top = max(0, min(rgb.height - 1, round(refined.top * rgb.height)))
        right = max(left + 1, min(rgb.width, round(refined.right * rgb.width)))
        bottom = max(top + 1, min(rgb.height, round(refined.bottom * rgb.height)))
        cropped = rgb.crop((left, top, right, bottom))
        output = BytesIO()
        cropped.save(output, format="JPEG", quality=95, optimize=True)
        return output.getvalue(), refined, cropped.width, cropped.height

    @staticmethod
    def _refine_red_box(pixels: NDArray[np.uint8], candidate: BoundingBox) -> BoundingBox:
        height, width, _ = pixels.shape
        left = max(0, min(width - 1, int(candidate.left * width)))
        top = max(0, min(height - 1, int(candidate.top * height)))
        right = max(left + 1, min(width, int(candidate.right * width)))
        bottom = max(top + 1, min(height, int(candidate.bottom * height)))
        region = pixels[top:bottom, left:right]
        red = region[:, :, 0].astype(np.int16)
        green = region[:, :, 1].astype(np.int16)
        blue = region[:, :, 2].astype(np.int16)
        mask = (red >= 110) & (red - green >= 35) & (red - blue >= 35)
        ys, xs = np.nonzero(mask)
        if xs.size < 50:
            return candidate

        red_left = left + float(np.percentile(xs, 2))
        red_right = left + float(np.percentile(xs, 98)) + 1.0
        red_top = top + float(np.percentile(ys, 2))
        red_bottom = top + float(np.percentile(ys, 98)) + 1.0
        red_width = red_right - red_left
        red_height = red_bottom - red_top
        candidate_width = right - left
        candidate_height = bottom - top
        if red_width < max(8.0, candidate_width * 0.25) or red_height < max(
            8.0, candidate_height * 0.25
        ):
            return candidate

        side = max(red_width, red_height) * 1.18
        center_x = (red_left + red_right) / 2.0
        center_y = (red_top + red_bottom) / 2.0
        refined_left = max(0.0, center_x - side / 2.0)
        refined_top = max(0.0, center_y - side / 2.0)
        refined_right = min(float(width), center_x + side / 2.0)
        refined_bottom = min(float(height), center_y + side / 2.0)
        return BoundingBox(
            left=refined_left / width,
            top=refined_top / height,
            right=refined_right / width,
            bottom=refined_bottom / height,
        )
