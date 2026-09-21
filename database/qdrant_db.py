import uuid
import asyncio
from typing import List, Tuple, Dict, Any, Optional
from langchain_core.documents import Document
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams, PointStruct, Filter
from config.settings import get_settings
from database.vector_store import VectorStoreProvider
import logging

logger = logging.getLogger('api_logger')

class QdrantProvider(VectorStoreProvider):
    def __init__(self):
        self.client: Optional[QdrantClient] = None
        self.settings = get_settings()

    def connect(self) -> None:
        try:
            self.client = QdrantClient(
                url=self.settings.QDRANT_URL,
                api_key=self.settings.QDRANT_API_KEY if self.settings.QDRANT_API_KEY else None
            )
            logger.info(f"Qdrant connected successfully at {self.settings.QDRANT_URL}")
        except Exception as e:
            logger.error(f"Failed to connect to Qdrant: {e}")
            raise

    def close(self) -> None:
        if self.client:
            self.client.close()

    def _ensure_collection(self, collection_name: str, dimension: int):
        if not self.client.collection_exists(collection_name):
            self.client.create_collection(
                collection_name=collection_name,
                vectors_config=VectorParams(size=dimension, distance=Distance.COSINE),
            )

    def _store_sync(self, collection_name: str, points: List[PointStruct]):
        self.client.upsert(
            collection_name=collection_name,
            points=points
        )

    def _resolve_point_id(self, chunk: Document) -> str:
        chunk_id = chunk.metadata.get("chunk_id")
        vector_type = chunk.metadata.get("vector_type")
        if chunk_id and not vector_type:
            return str(chunk_id)
        if chunk_id and vector_type:
            content_key = abs(hash(chunk.page_content)) % (10**12)
            return f"{chunk_id}:{vector_type}:{content_key}"
        parent_id = chunk.metadata.get("parent_chunk_id")
        if parent_id and vector_type:
            content_key = abs(hash(chunk.page_content)) % (10**12)
            return f"{parent_id}:{vector_type}:{content_key}"
        return str(uuid.uuid4())

    async def store_chunks(self, chunks: List[Document], embeddings: List[List[float]], collection_name: str = "rag_documents") -> bool:
        if not chunks or not embeddings:
            return False
            
        dim = len(embeddings[0])
        loop = asyncio.get_running_loop()
        
        await loop.run_in_executor(None, self._ensure_collection, collection_name, dim)
        
        points = []
        for chunk, embedding in zip(chunks, embeddings):
            stable_id = self._resolve_point_id(chunk)
            point_id = stable_id
            payload = {"page_content": chunk.page_content, **chunk.metadata}
            payload["embedding_version"] = chunk.metadata.get("embedding_version")
            points.append(
                PointStruct(
                    id=point_id,
                    vector=embedding,
                    payload={k: v for k, v in payload.items() if v is not None},
                )
            )
            
        try:
            await loop.run_in_executor(None, self._store_sync, collection_name, points)
            return True
        except Exception as e:
            logger.error(f"Failed to store chunks in Qdrant: {e}")
            return False

    def _search_sync(self, query_embedding: List[float], top_k: int, collection_name: str, qdrant_filter: Optional[Filter]) -> List[Any]:
        if not self.client.collection_exists(collection_name):
            return []
            
        return self.client.search(
            collection_name=collection_name,
            query_vector=query_embedding,
            limit=top_k,
            query_filter=qdrant_filter,
            with_payload=True
        )

    async def search(self, query_embedding: List[float], top_k: int = 4, collection_name: str = "rag_documents", filter_criteria: Optional[Dict[str, Any]] = None) -> List[Tuple[Document, float]]:
        loop = asyncio.get_running_loop()
        
        qdrant_filter = None
        if filter_criteria:
            from qdrant_client.http.models import FieldCondition, MatchValue
            must_conditions = []
            for k, v in filter_criteria.items():
                must_conditions.append(FieldCondition(key=k, match=MatchValue(value=v)))
            if must_conditions:
                qdrant_filter = Filter(must=must_conditions)

        try:
            hits = await loop.run_in_executor(None, self._search_sync, query_embedding, top_k, collection_name, qdrant_filter)
            
            results = []
            for hit in hits:
                payload = hit.payload or {}
                content = payload.pop("page_content", "")
                doc = Document(page_content=content, metadata=payload)
                results.append((doc, hit.score))
            return results
        except Exception as e:
            logger.error(f"Search failed in Qdrant: {e}")
            return []

    def _delete_sync(self, collection_name: str, document_id: str):
        from qdrant_client.http.models import FieldCondition, MatchValue, Filter
        if self.client.collection_exists(collection_name):
            self.client.delete(
                collection_name=collection_name,
                points_selector=Filter(
                    must=[
                        FieldCondition(
                            key="document_id",
                            match=MatchValue(value=document_id)
                        )
                    ]
                )
            )

    async def delete_by_document_id(self, document_id: str, collection_name: str = "rag_documents") -> bool:
        loop = asyncio.get_running_loop()
        try:
            await loop.run_in_executor(None, self._delete_sync, collection_name, document_id)
            return True
        except Exception as e:
            logger.error(f"Delete by document_id failed in Qdrant: {e}")
            return False
