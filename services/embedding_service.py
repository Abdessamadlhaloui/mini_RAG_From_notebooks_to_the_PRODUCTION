import logging
from typing import List, Optional

from config.settings import get_settings
from services.embedding_provider import EmbeddingProvider, build_embedding_provider

logger = logging.getLogger("api_logger")


class EmbeddingService:
    def __init__(self, provider: Optional[EmbeddingProvider] = None) -> None:
        self._settings = get_settings()
        self._provider = provider or build_embedding_provider()
        logger.info(
            "EmbeddingService initialized provider=%s version=%s dimension=%s",
            self._settings.EMBEDDING_PROVIDER,
            self.version,
            self.dimension,
        )

    @property
    def provider(self) -> EmbeddingProvider:
        return self._provider

    @property
    def version(self) -> str:
        return self._provider.version

    @property
    def dimension(self) -> int:
        return self._provider.dimension

    async def embed_documents(self, texts: List[str], batch_size: Optional[int] = None) -> List[List[float]]:
        if not texts:
            return []
        size = batch_size or self._settings.EMBEDDING_BATCH_SIZE
        if size < 1:
            raise ValueError("batch_size must be at least 1")
        all_embeddings: List[List[float]] = []
        for start in range(0, len(texts), size):
            batch = texts[start : start + size]
            logger.debug("Embedding document batch start=%s count=%s", start, len(batch))
            batch_vectors = await self._provider.embed_documents(batch)
            if len(batch_vectors) != len(batch):
                raise RuntimeError(
                    f"Embedding provider returned {len(batch_vectors)} vectors for {len(batch)} texts."
                )
            all_embeddings.extend(batch_vectors)
        return all_embeddings

    async def embed_query(self, text: str) -> List[float]:
        if not text or not text.strip():
            raise ValueError("Query text must be non-empty.")
        return await self._provider.embed_query(text.strip())

    async def get_embeddings(self, texts: List[str]) -> List[List[float]]:
        return await self.embed_documents(texts)

    async def get_query_embedding(self, query: str) -> List[float]:
        return await self.embed_query(query)
