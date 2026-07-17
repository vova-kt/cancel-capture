from __future__ import annotations

from dataclasses import dataclass

from cancel_capture.adapters.filesystem import ContentAddressedAssetStore
from cancel_capture.adapters.image import PillowImageProcessor
from cancel_capture.adapters.markdown_narratives import MarkdownNarrativeStore
from cancel_capture.adapters.metadata import BestEffortMetadataExtractor
from cancel_capture.adapters.openai_provider import (
    OpenAIClusterThemeProvider,
    OpenAICurrentNewsProvider,
    OpenAIEmbeddingProvider,
    OpenAINarrativeProvider,
    OpenAITextProvider,
    OpenAIVisionProvider,
)
from cancel_capture.adapters.sqlite_catalog import SQLiteCatalog
from cancel_capture.adapters.visual_embedding import PillowVisualEmbeddingProvider
from cancel_capture.application.cluster_theme import ClusterThemeService
from cancel_capture.application.ingest import IngestionService
from cancel_capture.application.narrative_experiment import NarrativeExperimentService
from cancel_capture.application.narrative_selection import NarrativeSelectionService
from cancel_capture.application.review import ReviewService
from cancel_capture.application.search import SearchService
from cancel_capture.application.visual_embeddings import VisualEmbeddingService
from cancel_capture.config import AppConfig
from cancel_capture.ports import EmbeddingProvider


@dataclass(frozen=True, slots=True)
class Services:
    config: AppConfig
    assets: ContentAddressedAssetStore
    catalog: SQLiteCatalog
    embeddings: EmbeddingProvider
    ingestion: IngestionService
    review: ReviewService
    search: SearchService


@dataclass(frozen=True, slots=True)
class NarrativeServices:
    selection: NarrativeSelectionService
    visual_embeddings: VisualEmbeddingService
    cluster_themes: ClusterThemeService
    experiments: NarrativeExperimentService
    store: MarkdownNarrativeStore


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
        embeddings=embeddings,
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


def build_narrative_services(services: Services) -> NarrativeServices:
    config = services.config
    images = PillowImageProcessor(
        max_image_pixels=config.storage.max_image_pixels,
        max_analysis_side=config.storage.max_analysis_side,
    )
    visual_provider = PillowVisualEmbeddingProvider(
        max_image_pixels=config.storage.max_image_pixels,
    )
    cluster_theme_provider = OpenAIClusterThemeProvider(config.narrative)
    current_news_provider = OpenAICurrentNewsProvider(config.narrative)
    narrative_provider = OpenAINarrativeProvider(config.narrative)
    store = MarkdownNarrativeStore(config.storage.data_dir)
    return NarrativeServices(
        selection=NarrativeSelectionService(services.catalog),
        visual_embeddings=VisualEmbeddingService(
            catalog=services.catalog,
            assets=services.assets,
            images=images,
            provider=visual_provider,
        ),
        cluster_themes=ClusterThemeService(cluster_theme_provider),
        experiments=NarrativeExperimentService(
            news=current_news_provider,
            narrative=narrative_provider,
            store=store,
        ),
        store=store,
    )
