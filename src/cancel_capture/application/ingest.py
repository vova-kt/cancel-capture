from __future__ import annotations

from dataclasses import replace
from uuid import uuid4

from cancel_capture.errors import ProviderResponseError, UnsupportedImageError
from cancel_capture.models import (
    BoundingBox,
    IngestionResult,
    IngestRequest,
    PhotoObservation,
    PreparedIngestion,
    PreparedSign,
    SignObservation,
)
from cancel_capture.ports import (
    AssetStore,
    CatalogRepository,
    EmbeddingProvider,
    ImageProcessor,
    MetadataExtractor,
    TextProvider,
    VisionProvider,
)


class IngestionService:
    def __init__(
        self,
        assets: AssetStore,
        images: ImageProcessor,
        metadata: MetadataExtractor,
        vision: VisionProvider,
        text: TextProvider,
        embeddings: EmbeddingProvider,
        catalog: CatalogRepository,
    ) -> None:
        self._assets = assets
        self._images = images
        self._metadata = metadata
        self._vision = vision
        self._text = text
        self._embeddings = embeddings
        self._catalog = catalog

    async def ingest(self, request: IngestRequest) -> IngestionResult:
        existing = self._catalog.find_ingestion(request.source_key)
        if existing is not None:
            return replace(existing, already_existed=True)
        if not request.data:
            raise UnsupportedImageError("The uploaded file is empty")

        original = self._assets.save_original(
            request.data,
            request.filename,
            request.declared_media_type,
        )
        original_path = self._assets.resolve(original.relative_path)
        extracted_metadata = self._metadata.extract(original_path)
        analysis_image = self._images.prepare(original_path)
        original = replace(
            original,
            width=analysis_image.source_width,
            height=analysis_image.source_height,
        )

        observation = await self._vision.inspect(analysis_image)
        if request.assume_sign and not observation.signs:
            observation = PhotoObservation(
                factual_summary=observation.factual_summary,
                signs=(
                    SignObservation(
                        ordinal=0,
                        box=BoundingBox.full_frame(),
                        confidence=0.5,
                        factual_summary=(
                            "The imported channel image is treated as an existing "
                            "prohibition sign. "
                            f"{observation.factual_summary}"
                        ),
                    ),
                ),
            )

        described = await self._text.describe(observation)
        if len(described.signs) != len(observation.signs):
            raise ProviderResponseError("Description count does not match detected signs")

        embedding_texts = (
            described.photo.search_text(),
            *(
                self._sign_search_text(
                    description.text.search_text(), description.topics_en, description.topics_ru
                )
                for description in described.signs
            ),
        )
        vectors = await self._embeddings.embed(embedding_texts)
        if len(vectors) != len(embedding_texts):
            raise ProviderResponseError("Embedding count does not match catalog documents")

        prepared_signs: list[PreparedSign] = []
        for observation_sign, description, embedding in zip(
            observation.signs, described.signs, vectors[1:], strict=True
        ):
            crop_data, crop_box, width, height = self._images.crop(
                original_path, analysis_image, observation_sign.box
            )
            crop_asset = self._assets.save_crop(crop_data, width, height)
            prepared_signs.append(
                PreparedSign(
                    item_id=uuid4().hex,
                    asset=crop_asset,
                    observation=observation_sign,
                    crop_box=crop_box,
                    description=description,
                    embedding=embedding,
                    status=request.initial_sign_status,
                    published_message=request.existing_channel_message,
                )
            )

        ingestion = PreparedIngestion(
            photo_item_id=uuid4().hex,
            source_kind=request.source_kind,
            source_key=request.source_key,
            asset=original,
            metadata=extracted_metadata,
            observation=observation,
            description=described.photo,
            embedding=vectors[0],
            vision_identity=self._vision.identity,
            text_identity=self._text.identity,
            signs=tuple(prepared_signs),
            source_message=request.source_message,
        )
        try:
            return self._catalog.insert_ingestion(ingestion)
        except Exception:
            concurrent = self._catalog.find_ingestion(request.source_key)
            if concurrent is not None:
                return replace(concurrent, already_existed=True)
            raise

    @staticmethod
    def _sign_search_text(
        bilingual: str, topics_en: tuple[str, ...], topics_ru: tuple[str, ...]
    ) -> str:
        english = ", ".join(topics_en)
        russian = ", ".join(topics_ru)
        return f"{bilingual}\n\nTopics (English): {english}\nТемы (Русский): {russian}"
