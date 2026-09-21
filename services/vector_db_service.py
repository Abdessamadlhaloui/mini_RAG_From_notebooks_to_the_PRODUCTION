from database.vector_store import VectorDatabase
from langchain_core.documents import Document
from typing import Any, Dict, List, Optional, Tuple

class VectorDBService:
    def __init__(self, collection_name: str = 'rag_documents') -> None:
        self.collection_name = collection_name
        
    async def store_chunks(self, chunks: List[Document], embeddings: List[List[float]]) -> bool:
        provider = VectorDatabase.get_provider()
        return await provider.store_chunks(chunks, embeddings, self.collection_name)
        
    async def search(
        self,
        query_embedding: List[float],
        top_k: int = 4,
        filter_criteria: Optional[Dict[str, Any]] = None,
    ) -> List[Tuple[Document, float]]:
        provider = VectorDatabase.get_provider()
        return await provider.search(query_embedding, top_k, self.collection_name, filter_criteria)
