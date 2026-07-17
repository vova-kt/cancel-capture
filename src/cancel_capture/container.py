from __future__ import annotations

from dataclasses import dataclass

from cancel_capture.adapters.filesystem import ContentAddressedAssetStore
from cancel_capture.adapters.image import PillowImageProcessor
from cancel_capture.adapters.metadata import BestEffortMetadataExtractor
from cancel_capture.adapters.openai_provider import (
    OpenAIEmbeddingProvider,
    OpenAITextProvider,
    OpenAIVisionProvider,
)
from cancel_capture.adapters.sqlite_catalog import SQLiteCatalog
from cancel_capture.application.ingest import IngestionService
from cancel_capture.application.review import ReviewService
from cancel_capture.application.search import SearchService
from cancel_capture.config import AppConfig


@dataclass(frozen=True, slots=True)
class Services:
    config: AppConfig
    assets: ContentAddressedAssetStore
    catalog: SQLiteCatalog
    ingestion: IngestionService
    review: ReviewService
    search: SearchService


def build_services(config: AppConfig) -> Services:
    assets = ContentAddressedAssetStore(
        root=config.storage.data_dir,
        max_upload_bytes=config.storage.max_upload_bytes,
    )
    catalog = SQLiteCatalog(config.storage.sqlite_path)
    catalog.initialize()
    images = PillowImageProcessor(
        max_image_pixels=config.storage.max_image_pixels,
        max_analysis_side=config.storage.max_analysis_side,
    )
    vision = OpenAIVisionProvider(config.vision)
    text = OpenAITextProvider(config.text)
    embeddings = OpenAIEmbeddingProvider(config.embedding)
    return Services(
        config=config,
        assets=assets,
        catalog=catalog,
        ingestion=IngestionService(
            assets=assets,
            images=images,
            metadata=BestEffortMetadataExtractor(),
            vision=vision,
            text=text,
            embeddings=embeddings,
            catalog=catalog,
        ),
        review=ReviewService(catalog),
        search=SearchService(catalog, embeddings, assets),
    )
