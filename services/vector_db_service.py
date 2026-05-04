"""
VectorDBService — manages ChromaDB collection operations.

All synchronous ChromaDB calls (add, query) are dispatched to a
thread-pool executor to avoid blocking the async event loop.
"""
import uuid
import asyncio
from functools import partial
from database.vector_db import ChromaDBClient
from langchain.docstore.document import Document
from typing import List, Tuple


class VectorDBService:
    """Handles storage and retrieval of document embeddings in ChromaDB."""

    def __init__(self, collection_name: str = "rag_documents") -> None:
        client = ChromaDBClient.get_client()
        self.collection = client.get_or_create_collection(name=collection_name)

    def _add_sync(
        self,
        ids: List[str],
        embeddings: List[List[float]],
        documents: List[str],
        metadatas: List[dict],
    ) -> None:
        """Synchronous ChromaDB add — runs inside an executor."""
        self.collection.add(
            embeddings=embeddings,
            documents=documents,
            metadatas=metadatas,
            ids=ids,
        )

    def _query_sync(
        self, query_embedding: List[float], top_k: int
    ) -> dict:
        """Synchronous ChromaDB query — runs inside an executor."""
        return self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            include=["documents", "metadatas", "distances"],
        )

    async def store_chunks(
        self, chunks: List[Document], embeddings: List[List[float]]
    ) -> bool:
        """
        Stores document chunks and their embeddings into ChromaDB.
        Returns True on success, False if there is nothing to store.
        """
        if not chunks:
            return False

        ids = [str(uuid.uuid4()) for _ in chunks]
        documents = [chunk.page_content for chunk in chunks]
        metadatas = [chunk.metadata for chunk in chunks]

        loop = asyncio.get_running_loop()
        await loop.run_in_executor(
            None, partial(self._add_sync, ids, embeddings, documents, metadatas)
        )
        return True

    async def search(
        self, query_embedding: List[float], top_k: int = 4
    ) -> List[Tuple[Document, float]]:
        """
        Performs a similarity search and returns (Document, distance) tuples.
        Lower distance = higher relevance.
        """
        loop = asyncio.get_running_loop()
        results = await loop.run_in_executor(
            None, partial(self._query_sync, query_embedding, top_k)
        )

        docs_with_scores: List[Tuple[Document, float]] = []
        if results and results.get("documents") and len(results["documents"]) > 0:
            for i in range(len(results["documents"][0])):
                doc_text = results["documents"][0][i]
                metadata = (
                    results["metadatas"][0][i] if results.get("metadatas") else {}
                )
                score = (
                    results["distances"][0][i] if results.get("distances") else 0.0
                )
                doc = Document(page_content=doc_text, metadata=metadata)
                docs_with_scores.append((doc, score))

        return docs_with_scores
