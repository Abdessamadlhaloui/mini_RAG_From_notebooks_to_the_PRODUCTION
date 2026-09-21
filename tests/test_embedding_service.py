import pytest
from typing import List

from services.embedding_service import EmbeddingService
from services.embedding_provider import EmbeddingProvider, build_embedding_provider


class _FakeEmbeddingProvider(EmbeddingProvider):
    def __init__(self, dimension: int = 8) -> None:
        self._dimension = dimension
        self.document_calls: List[List[str]] = []
        self.query_calls: List[str] = []

    @property
    def version(self) -> str:
        return "fake_v1"

    @property
    def dimension(self) -> int:
        return self._dimension

    async def embed_documents(self, texts: List[str]) -> List[List[float]]:
        self.document_calls.append(list(texts))
        return [[float(i + 1)] * self._dimension for i, _ in enumerate(texts)]

    async def embed_query(self, text: str) -> List[float]:
        self.query_calls.append(text)
        return [0.25] * self._dimension


@pytest.mark.asyncio
async def test_embed_documents_batches():
    provider = _FakeEmbeddingProvider()
    service = EmbeddingService(provider=provider)
    texts = [f"t{i}" for i in range(5)]
    vectors = await service.embed_documents(texts, batch_size=2)
    assert len(vectors) == 5
    assert len(provider.document_calls) == 3
    assert provider.document_calls[0] == ["t0", "t1"]
    assert provider.document_calls[-1] == ["t4"]


@pytest.mark.asyncio
async def test_backward_compatible_aliases():
    provider = _FakeEmbeddingProvider()
    service = EmbeddingService(provider=provider)
    doc_vectors = await service.get_embeddings(["hello"])
    query_vector = await service.get_query_embedding("hello")
    assert len(doc_vectors) == 1
    assert len(query_vector) == provider.dimension


@pytest.mark.asyncio
async def test_embed_query_rejects_blank():
    service = EmbeddingService(provider=_FakeEmbeddingProvider())
    with pytest.raises(ValueError):
        await service.embed_query("   ")


def test_build_embedding_provider_openai(monkeypatch):
    monkeypatch.setenv("EMBEDDING_PROVIDER", "openai")
    from config.settings import get_settings

    get_settings.cache_clear()
    provider = build_embedding_provider("openai")
    assert provider.dimension >= 1
    get_settings.cache_clear()


def test_build_embedding_provider_unknown():
    with pytest.raises(ValueError):
        build_embedding_provider("unknown-provider")
