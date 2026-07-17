from __future__ import annotations

import pytest

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
from cancel_capture.provider_registry import (
    build_cluster_theme,
    build_current_news,
    build_embedding,
    build_narrative,
    build_text,
    build_vision,
)


def _config(provider: str = "openai") -> ProviderConfig:
    return ProviderConfig(
        provider=provider,
        api_key="unused",
        base_url=None,
        model="stub-model",
        identity_namespace="test",
    )


def test_registry_dispatches_openai_role_by_config_provider() -> None:
    config = _config()
    assert isinstance(build_vision(config), OpenAIVisionProvider)
    assert isinstance(build_text(config), OpenAITextProvider)
    assert isinstance(build_embedding(config), OpenAIEmbeddingProvider)
    assert isinstance(build_narrative(config), OpenAINarrativeProvider)
    assert isinstance(build_cluster_theme(config), OpenAIClusterThemeProvider)
    assert isinstance(build_current_news(config), OpenAICurrentNewsProvider)


def test_registry_reports_unknown_provider_and_lists_available_names() -> None:
    config = _config(provider="mystery-corp")
    with pytest.raises(ConfigurationError, match="vision") as excinfo:
        build_vision(config)
    assert "openai" in str(excinfo.value)
    assert "mystery-corp" in str(excinfo.value)
