import pytest
from unittest.mock import AsyncMock, MagicMock
from langchain_core.documents import Document

from workers.celery_worker import _process_reindex_batch


@pytest.mark.asyncio
async def test_process_reindex_batch_updates_mongo_and_vector_store():
    embed_service = MagicMock()
    embed_service.embed_documents = AsyncMock(return_value=[[0.1, 0.2]])
    vector_db = MagicMock()
    vector_db.store_chunks = AsyncMock(return_value=True)
    db = MagicMock()
    db.__getitem__ = MagicMock(return_value=MagicMock())
    chunks_col = db["chunks"]
    chunks_col.update_one = AsyncMock()

    batch = [
        (
            "chunk-1",
            Document(page_content="hello", metadata={"chunk_id": "chunk-1", "embedding_version": "v2"}),
        )
    ]
    processed = await _process_reindex_batch(batch, embed_service, vector_db, db, "v2")

    embed_service.embed_documents.assert_awaited_once_with(["hello"])
    vector_db.store_chunks.assert_awaited_once()
    chunks_col.update_one.assert_awaited_once_with({"_id": "chunk-1"}, {"$set": {"embedding_version": "v2"}})
    assert processed == 1


@pytest.mark.asyncio
async def test_process_reindex_batch_raises_when_vector_store_rejects():
    embed_service = MagicMock()
    embed_service.embed_documents = AsyncMock(return_value=[[0.1, 0.2]])
    vector_db = MagicMock()
    vector_db.store_chunks = AsyncMock(return_value=False)
    db = MagicMock()

    batch = [
        (
            "chunk-1",
            Document(page_content="hello", metadata={"chunk_id": "chunk-1", "embedding_version": "v2"}),
        )
    ]
    with pytest.raises(RuntimeError):
        await _process_reindex_batch(batch, embed_service, vector_db, db, "v2")
