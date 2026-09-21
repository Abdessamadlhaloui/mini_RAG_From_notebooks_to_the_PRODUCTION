from typing import List
from unittest.mock import AsyncMock, MagicMock

import pytest
from langchain_core.documents import Document

from database.mongo_db import MongoDBClient
from services.embedding_indexing_service import EmbeddingIndexingService
from services.embedding_provider import EmbeddingProvider
from services.embedding_service import EmbeddingService


class _FakeEmbeddingProvider(EmbeddingProvider):
    @property
    def version(self) -> str:
        return "fake_v2"

    @property
    def dimension(self) -> int:
        return 2

    async def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return [[1.0, 2.0] for _ in texts]

    async def embed_query(self, text: str) -> List[float]:
        return [1.0, 2.0]


@pytest.mark.asyncio
async def test_index_chunks_sets_version_and_persists_chunks(monkeypatch):
    chunks_collection = MagicMock()
    chunks_collection.update_one = AsyncMock()
    fake_db = {"chunks": chunks_collection}
    monkeypatch.setattr(MongoDBClient, "get_db", classmethod(lambda cls: fake_db))
    vector_db = MagicMock()
    vector_db.store_chunks = AsyncMock(return_value=True)
    service = EmbeddingIndexingService(
        embedding_service=EmbeddingService(provider=_FakeEmbeddingProvider()),
        vector_db_service=vector_db,
    )

    result = await service.index_chunks(
        document_id="doc-1",
        filename="paper.txt",
        chunks=[Document(page_content="hello", metadata={"page": 3})],
    )

    assert result.document_id == "doc-1"
    assert result.embedding_version == "fake_v2"
    assert result.chunks_indexed == 1
    stored_chunks = vector_db.store_chunks.await_args.args[0]
    assert stored_chunks[0].metadata["embedding_version"] == "fake_v2"
    assert stored_chunks[0].metadata["document_id"] == "doc-1"
    chunks_collection.update_one.assert_awaited_once()


@pytest.mark.asyncio
async def test_index_chunks_raises_when_vector_store_rejects(monkeypatch):
    chunks_collection = MagicMock()
    chunks_collection.update_one = AsyncMock()
    fake_db = {"chunks": chunks_collection}
    monkeypatch.setattr(MongoDBClient, "get_db", classmethod(lambda cls: fake_db))
    vector_db = MagicMock()
    vector_db.store_chunks = AsyncMock(return_value=False)
    service = EmbeddingIndexingService(
        embedding_service=EmbeddingService(provider=_FakeEmbeddingProvider()),
        vector_db_service=vector_db,
    )

    with pytest.raises(RuntimeError):
        await service.index_chunks(
            document_id="doc-1",
            filename="paper.txt",
            chunks=[Document(page_content="hello", metadata={})],
        )
    chunks_collection.update_one.assert_not_called()
