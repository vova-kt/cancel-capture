from __future__ import annotations

from collections.abc import Callable

from cancel_capture.adapters.openai_provider import (
    OpenAIClusterThemeProvider,
    OpenAICurrentNewsProvider,
    OpenAIEmbeddingProvider,
    OpenAINarrativeProvider,
    OpenAITextProvider,
    OpenAIVisionProvider,
)
from cancel_capture.config import ProviderConfig
from cancel_capture.errors import ConfigurationError
from cancel_capture.ports import (
    ClusterThemeProvider,
    CurrentNewsProvider,
    EmbeddingProvider,
    NarrativeProvider,
    TextProvider,
    VisionProvider,
)

# Adding a new backend for any role is a one-liner: register the class under the ProviderConfig
# name so the config key (e.g. VISION_PROVIDER=azure) routes to the matching adapter.
VISION_PROVIDERS: dict[str, Callable[[ProviderConfig], VisionProvider]] = {
    "openai": OpenAIVisionProvider,
}

TEXT_PROVIDERS: dict[str, Callable[[ProviderConfig], TextProvider]] = {
    "openai": OpenAITextProvider,
}

EMBEDDING_PROVIDERS: dict[str, Callable[[ProviderConfig], EmbeddingProvider]] = {
    "openai": OpenAIEmbeddingProvider,
}

NARRATIVE_PROVIDERS: dict[str, Callable[[ProviderConfig], NarrativeProvider]] = {
    "openai": OpenAINarrativeProvider,
}

CLUSTER_THEME_PROVIDERS: dict[str, Callable[[ProviderConfig], ClusterThemeProvider]] = {
    "openai": OpenAIClusterThemeProvider,
}

CURRENT_NEWS_PROVIDERS: dict[str, Callable[[ProviderConfig], CurrentNewsProvider]] = {
    "openai": OpenAICurrentNewsProvider,
}


def build_vision(config: ProviderConfig) -> VisionProvider:
    return _build("vision", config, VISION_PROVIDERS)


def build_text(config: ProviderConfig) -> TextProvider:
    return _build("text", config, TEXT_PROVIDERS)


def build_embedding(config: ProviderConfig) -> EmbeddingProvider:
    return _build("embedding", config, EMBEDDING_PROVIDERS)


def build_narrative(config: ProviderConfig) -> NarrativeProvider:
    return _build("narrative", config, NARRATIVE_PROVIDERS)


def build_cluster_theme(config: ProviderConfig) -> ClusterThemeProvider:
    return _build("cluster-theme", config, CLUSTER_THEME_PROVIDERS)


def build_current_news(config: ProviderConfig) -> CurrentNewsProvider:
    return _build("current-news", config, CURRENT_NEWS_PROVIDERS)


def _build[T](
    role: str,
    config: ProviderConfig,
    registry: dict[str, Callable[[ProviderConfig], T]],
) -> T:
    factory = registry.get(config.provider)
    if factory is None:
        available = ", ".join(sorted(registry)) or "<none>"
        raise ConfigurationError(
            f"No {role} provider registered for {config.provider!r}; available: {available}"
        )
    return factory(config)
