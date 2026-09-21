from langchain_core.documents import Document

from database.qdrant_db import QdrantProvider


def test_resolve_point_id_uses_chunk_id_for_primary_vector():
    provider = QdrantProvider()
    doc = Document(page_content="body", metadata={"chunk_id": "abc-123"})
    assert provider._resolve_point_id(doc) == "abc-123"


def test_resolve_point_id_distinguishes_multi_vector_types():
    provider = QdrantProvider()
    summary = Document(
        page_content="summary text",
        metadata={"chunk_id": "abc-123", "vector_type": "summary"},
    )
    question = Document(
        page_content="What is RAG?",
        metadata={"chunk_id": "abc-123", "vector_type": "question"},
    )
    assert provider._resolve_point_id(summary) != provider._resolve_point_id(question)
