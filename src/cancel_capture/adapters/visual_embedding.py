# pyright: reportMissingTypeStubs=false

from __future__ import annotations

import warnings
from io import BytesIO

import numpy as np
from numpy.typing import NDArray
from PIL import Image, ImageOps, UnidentifiedImageError

from cancel_capture.errors import ImageTooLargeError, UnsupportedImageError
from cancel_capture.models import Embedding, PreparedImage, ProviderIdentity


class PillowVisualEmbeddingProvider:
    def __init__(
        self,
        *,
        max_image_pixels: int,
        grid_size: int = 24,
        histogram_bins: int = 12,
    ) -> None:
        if max_image_pixels <= 0:
            raise ValueError("Maximum image pixels must be positive")
        if grid_size < 4:
            raise ValueError("Visual embedding grid size must be at least 4")
        if histogram_bins < 2:
            raise ValueError("Visual embedding histogram must have at least 2 bins")
        self._max_image_pixels = max_image_pixels
        self._grid_size = grid_size
        self._histogram_bins = histogram_bins

    @property
    def identity(self) -> ProviderIdentity:
        return ProviderIdentity(
            provider="local",
            model=(f"pillow-visual-v1-grid-{self._grid_size}-bins-{self._histogram_bins}"),
            namespace="deterministic-handcrafted",
        )

    @property
    def dimensions(self) -> int:
        spatial_dimensions = self._grid_size * self._grid_size * 3
        histogram_dimensions = self._histogram_bins * 3
        return spatial_dimensions + histogram_dimensions + 2

    def embed(self, images: tuple[PreparedImage, ...]) -> tuple[Embedding, ...]:
        return tuple(self.embed_one(image) for image in images)

    def embed_one(self, image: PreparedImage) -> Embedding:
        pixels = self._load_pixels(image)
        feature_blocks = self._feature_blocks(pixels, image.source_width, image.source_height)
        combined = np.concatenate(tuple(self._normalize(block) for block in feature_blocks))
        normalized = self._normalize(combined)
        return Embedding(
            identity=self.identity,
            values=tuple(float(value) for value in normalized),
        )

    def _load_pixels(self, image: PreparedImage) -> NDArray[np.float64]:
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("error", Image.DecompressionBombWarning)
                with Image.open(BytesIO(image.data)) as opened:
                    if opened.width * opened.height > self._max_image_pixels:
                        raise ImageTooLargeError(
                            f"Decoded image has {opened.width * opened.height} pixels; "
                            f"limit is {self._max_image_pixels}"
                        )
                    oriented = ImageOps.exif_transpose(opened)
                    rgb = self._on_white(oriented)
                    rgb.thumbnail(
                        (self._grid_size, self._grid_size),
                        Image.Resampling.LANCZOS,
                    )
                    canvas = Image.new("RGB", (self._grid_size, self._grid_size), (255, 255, 255))
                    offset = (
                        (self._grid_size - rgb.width) // 2,
                        (self._grid_size - rgb.height) // 2,
                    )
                    canvas.paste(rgb, offset)
                    pixels = np.asarray(canvas, dtype=np.float64) / 255.0
                    return pixels
        except (UnidentifiedImageError, OSError) as error:
            raise UnsupportedImageError("The file is not a supported image") from error
        except Image.DecompressionBombWarning as error:
            raise ImageTooLargeError("The image exceeds Pillow's safe pixel limit") from error

    @staticmethod
    def _on_white(image: Image.Image) -> Image.Image:
        if "A" not in image.getbands():
            return image.convert("RGB")
        rgba = image.convert("RGBA")
        background = Image.new("RGBA", rgba.size, (255, 255, 255, 255))
        background.alpha_composite(rgba)
        return background.convert("RGB")

    def _feature_blocks(
        self,
        pixels: NDArray[np.float64],
        source_width: int,
        source_height: int,
    ) -> tuple[NDArray[np.float64], ...]:
        red = pixels[:, :, 0]
        green = pixels[:, :, 1]
        blue = pixels[:, :, 2]
        gray = red * 0.299 + green * 0.587 + blue * 0.114
        darkness = 1.0 - gray
        red_dominance = np.clip(red - np.maximum(green, blue), 0.0, 1.0)

        horizontal = np.zeros_like(gray)
        vertical = np.zeros_like(gray)
        horizontal[:, 1:] = np.abs(np.diff(gray, axis=1))
        vertical[1:, :] = np.abs(np.diff(gray, axis=0))
        edges = np.sqrt(horizontal * horizontal + vertical * vertical)

        histograms: list[NDArray[np.float64]] = []
        for channel in (red, green, blue):
            counts, _edges = np.histogram(
                channel,
                bins=self._histogram_bins,
                range=(0.0, 1.0),
            )
            histograms.append(counts.astype(np.float64))
        color_histogram = np.concatenate(tuple(histograms))
        aspect = np.asarray(
            (
                source_width / max(source_width, source_height),
                source_height / max(source_width, source_height),
            ),
            dtype=np.float64,
        )
        return (
            darkness.reshape(-1),
            red_dominance.reshape(-1),
            edges.reshape(-1),
            color_histogram,
            aspect,
        )

    @staticmethod
    def _normalize(values: NDArray[np.float64]) -> NDArray[np.float64]:
        norm = float(np.linalg.norm(values))
        if norm == 0.0:
            return values.copy()
        return values / norm
