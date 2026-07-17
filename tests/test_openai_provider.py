from types import SimpleNamespace

import pytest

from cancel_capture.adapters.openai_provider import OpenAIEmbeddingProvider
from cancel_capture.config import ProviderConfig
from cancel_capture.errors import ProviderResponseError


class FakeEmbeddingsEndpoint:
    def __init__(self, indexes: tuple[int, ...]) -> None:
        self.indexes = indexes
        self.request: dict[str, object] | None = None

    async def create(self, **kwargs):
        self.request = kwargs
        return SimpleNamespace(
            data=[
                SimpleNamespace(index=index, embedding=[float(index), 1.0])
                for index in self.indexes
            ]
        )


class FakeOpenAIClient:
    def __init__(self, endpoint: FakeEmbeddingsEndpoint) -> None:
        self.embeddings = endpoint


def _provider(indexes: tuple[int, ...]) -> tuple[OpenAIEmbeddingProvider, FakeEmbeddingsEndpoint]:
    endpoint = FakeEmbeddingsEndpoint(indexes)
    provider = object.__new__(OpenAIEmbeddingProvider)
    provider._config = ProviderConfig(
        provider="openai",
        api_key="unused",
        base_url=None,
        model="embedding-test",
        identity_namespace="test-deployment",
        dimensions=2,
    )
    provider._client = FakeOpenAIClient(endpoint)
    return provider, endpoint


async def test_embedding_adapter_preserves_text_order_and_request_identity() -> None:
    provider, endpoint = _provider((1, 0))

    vectors = await provider.embed(("first", "second"))

    assert vectors[0].values == (0.0, 1.0)
    assert vectors[0].identity.namespace == "test-deployment"
    assert endpoint.request is not None
    assert endpoint.request["input"] == ["first", "second"]
    assert endpoint.request["dimensions"] == 2


async def test_embedding_adapter_rejects_duplicate_response_indexes() -> None:
    provider, _endpoint = _provider((0, 0))

    with pytest.raises(ProviderResponseError, match="indexes"):
        await provider.embed(("first", "second"))
