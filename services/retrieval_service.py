"""
RetrievalService — embeds a query and retrieves relevant context from the vector DB.
Results are reranked by distance (ascending = most relevant first).
"""
from services.embedding_service import EmbeddingService
from services.vector_db_service import VectorDBService
from langchain.docstore.document import Document
from typing import List, Tuple


class RetrievalService:
    """Orchestrates query embedding and vector-DB retrieval."""

    def __init__(
        self,
        embedding_service: EmbeddingService | None = None,
        vector_db_service: VectorDBService | None = None,
    ) -> None:
        self.embedding_service = embedding_service or EmbeddingService()
        self.vector_db = vector_db_service or VectorDBService()

    async def retrieve_context(
        self, query: str, top_k: int
    ) -> List[Tuple[Document, float]]:
        """
        Embeds the query, searches the vector DB, and returns results
        sorted by ascending distance (lower = more relevant).
        """
        # 1. Embed query
        query_embedding = await self.embedding_service.get_query_embedding(query)

        # 2. Retrieve from Vector DB
        docs_with_scores = await self.vector_db.search(query_embedding, top_k=top_k)

        # 3. Rerank by distance (ascending)
        docs_with_scores.sort(key=lambda x: x[1])

        return docs_with_scores
