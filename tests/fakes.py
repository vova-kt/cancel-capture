from __future__ import annotations

from dataclasses import dataclass
from io import BytesIO
from pathlib import Path

from PIL import Image, ImageDraw

from cancel_capture.adapters.filesystem import ContentAddressedAssetStore
from cancel_capture.adapters.image import PillowImageProcessor
from cancel_capture.adapters.sqlite_catalog import SQLiteCatalog
from cancel_capture.application.ingest import IngestionService
from cancel_capture.application.review import ReviewService
from cancel_capture.application.search import SearchService
from cancel_capture.models import (
    BilingualText,
    DescribedPhoto,
    Embedding,
    ImageMetadata,
    PhotoObservation,
    PreparedImage,
    ProviderIdentity,
    SignDescription,
)


class StaticVision:
    def __init__(self, observation: PhotoObservation) -> None:
        self.observation = observation
        self.calls = 0

    @property
    def identity(self) -> ProviderIdentity:
        return ProviderIdentity("fake", "vision-v1")

    async def inspect(self, image: PreparedImage) -> PhotoObservation:
        assert image.media_type == "image/jpeg"
        assert image.data.startswith(b"\xff\xd8")
        self.calls += 1
        return self.observation


class StaticText:
    def __init__(self) -> None:
        self.calls = 0

    @property
    def identity(self) -> ProviderIdentity:
        return ProviderIdentity("fake", "text-v1")

    async def describe(self, observation: PhotoObservation) -> DescribedPhoto:
        self.calls += 1
        signs: list[SignDescription] = []
        for sign in observation.signs:
            lowered = sign.factual_summary.casefold()
            if "bicycle" in lowered:
                topics_en = ("bicycle", "mobility")
                topics_ru = ("велосипед", "мобильность")
            elif "dog" in lowered:
                topics_en = ("dog", "animal")
                topics_ru = ("собака", "животное")
            else:
                topics_en = ("prohibition",)
                topics_ru = ("запрет",)
            signs.append(
                SignDescription(
                    ordinal=sign.ordinal,
                    text=BilingualText(
                        en=f"Sign: {sign.factual_summary}",
                        ru=f"Знак: {sign.factual_summary}",
                    ),
                    topics_en=topics_en,
                    topics_ru=topics_ru,
                )
            )
        return DescribedPhoto(
            photo=BilingualText(
                en=f"Photo: {observation.factual_summary}",
                ru=f"Фото: {observation.factual_summary}",
            ),
            signs=tuple(signs),
        )


class KeywordEmbeddings:
    def __init__(self) -> None:
        self.calls = 0

    @property
    def identity(self) -> ProviderIdentity:
        return ProviderIdentity("fake", "embedding-v1")

    async def embed(self, texts: tuple[str, ...]) -> tuple[Embedding, ...]:
        self.calls += 1
        return tuple(Embedding(self.identity, self._vector(text)) for text in texts)

    @staticmethod
    def _vector(text: str) -> tuple[float, ...]:
        lowered = text.casefold()
        bicycle = 1.0 if "bicycle" in lowered or "велосипед" in lowered else 0.05
        dog = 1.0 if "dog" in lowered or "собак" in lowered else 0.05
        scene = 1.0 if "street" in lowered or "улиц" in lowered else 0.1
        return bicycle, dog, scene


class StaticMetadata:
    def extract(self, path: Path) -> ImageMetadata:
        assert path.exists()
        return ImageMetadata(
            raw_json='{"EXIF:Make":"Test Camera","EXIF:SerialNumber":"private"}',
            captured_at="2026:07:17 10:30:00",
            latitude=52.5,
            longitude=13.4,
            camera_make="Test Camera",
            extractor="fake",
        )


@dataclass(slots=True)
class TestStack:
    assets: ContentAddressedAssetStore
    catalog: SQLiteCatalog
    ingestion: IngestionService
    review: ReviewService
    search: SearchService
    vision: StaticVision
    text: StaticText
    embeddings: KeywordEmbeddings


def build_stack(
    tmp_path: Path, observation: PhotoObservation, max_analysis_side: int = 2048
) -> TestStack:
    assets = ContentAddressedAssetStore(tmp_path / "data", max_upload_bytes=10_000_000)
    catalog = SQLiteCatalog(tmp_path / "data" / "catalog.sqlite3")
    catalog.initialize()
    vision = StaticVision(observation)
    text = StaticText()
    embeddings = KeywordEmbeddings()
    ingestion = IngestionService(
        assets=assets,
        images=PillowImageProcessor(
            max_image_pixels=10_000_000, max_analysis_side=max_analysis_side
        ),
        metadata=StaticMetadata(),
        vision=vision,
        text=text,
        embeddings=embeddings,
        catalog=catalog,
    )
    return TestStack(
        assets=assets,
        catalog=catalog,
        ingestion=ingestion,
        review=ReviewService(catalog),
        search=SearchService(catalog, embeddings, assets),
        vision=vision,
        text=text,
        embeddings=embeddings,
    )


def sample_jpeg() -> bytes:
    image = Image.new("RGB", (600, 400), "white")
    draw = ImageDraw.Draw(image)
    draw.ellipse((70, 70, 250, 250), outline="red", width=18)
    draw.line((100, 220, 220, 100), fill="red", width=18)
    draw.ellipse((350, 100, 510, 260), outline="red", width=18)
    draw.line((375, 235, 485, 125), fill="red", width=18)
    output = BytesIO()
    image.save(output, format="JPEG", quality=95)
    return output.getvalue()
