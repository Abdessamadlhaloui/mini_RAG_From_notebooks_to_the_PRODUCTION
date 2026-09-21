from abc import ABC, abstractmethod
from typing import List, Tuple, Dict, Any, Optional
from langchain_core.documents import Document

class VectorStoreProvider(ABC):
    
    @abstractmethod
    def connect(self) -> None:
        pass
        
    @abstractmethod
    def close(self) -> None:
        pass
        
    @abstractmethod
    async def store_chunks(self, chunks: List[Document], embeddings: List[List[float]], collection_name: str = "rag_documents") -> bool:
        pass
        
    @abstractmethod
    async def search(self, query_embedding: List[float], top_k: int = 4, collection_name: str = "rag_documents", filter_criteria: Optional[Dict[str, Any]] = None) -> List[Tuple[Document, float]]:
        pass

    @abstractmethod
    async def delete_by_document_id(self, document_id: str, collection_name: str = "rag_documents") -> bool:
        pass

class VectorDatabase:
    _provider: Optional[VectorStoreProvider] = None

    @classmethod
    def initialize(cls, provider: VectorStoreProvider) -> None:
        cls._provider = provider
        cls._provider.connect()

    @classmethod
    def get_provider(cls) -> VectorStoreProvider:
        if cls._provider is None:
            raise RuntimeError("VectorDatabase provider not initialized.")
        return cls._provider
