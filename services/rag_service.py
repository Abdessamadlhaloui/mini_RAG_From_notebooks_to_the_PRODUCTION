"""
RagService — orchestrates the full RAG pipeline:
    query → sanitize → embed → retrieve → rerank → generate → respond

Includes an in-memory cache for identical queries to reduce redundant
LLM calls. In production, replace ``_cache`` with Redis.
"""
import hashlib
from collections import OrderedDict
from services.retrieval_service import RetrievalService
from services.generation_service import GenerationService
from helpers.text_helper import sanitize_query
from schemas.rag_schema import QueryRequest, QueryResponse, SourceDocument
from config.settings import get_settings
import logging

logger = logging.getLogger("api_logger")

# ---------------------------------------------------------------------------
# Simple LRU-style in-memory cache (swap with Redis in production at scale)
# ---------------------------------------------------------------------------
_CACHE_MAX_SIZE = 128
_cache: OrderedDict[str, QueryResponse] = OrderedDict()


def _cache_key(query: str, top_k: int) -> str:
    """Produces a deterministic hash for a query + top_k pair."""
    raw = f"{query.strip().lower()}::{top_k}"
    return hashlib.sha256(raw.encode()).hexdigest()


class RagService:
    """End-to-end RAG pipeline with caching and input sanitization."""

    def __init__(
        self,
        retrieval_service: RetrievalService | None = None,
        generation_service: GenerationService | None = None,
    ) -> None:
        self.retrieval_service = retrieval_service or RetrievalService()
        self.generation_service = generation_service or GenerationService()
        self.settings = get_settings()

    async def run_pipeline(self, request: QueryRequest) -> QueryResponse:
        """Runs the full RAG pipeline for a given query."""
        top_k = request.top_k or self.settings.top_k

        # --- Input sanitization ---
        clean_query = sanitize_query(request.query)

        # --- Cache lookup ---
        key = _cache_key(clean_query, top_k)
        if key in _cache:
            logger.info("Cache HIT for query key %s", key[:12])
            cached = _cache[key]
            # Return a copy with the cached flag set
            return QueryResponse(
                answer=cached.answer,
                sources=cached.sources,
                cached=True,
            )

        # --- 1. Retrieve & Rerank ---
        docs_with_scores = await self.retrieval_service.retrieve_context(
            clean_query, top_k
        )

        if not docs_with_scores:
            return QueryResponse(
                answer="I couldn't find any relevant context in the database to answer your question.",
                sources=[],
            )

        # --- 2. Build context string ---
        context_texts = [doc.page_content for doc, _ in docs_with_scores]
        context_string = "\n\n---\n\n".join(context_texts)

        # --- 3. Generate answer ---
        answer = await self.generation_service.generate_answer(
            clean_query, context_string
        )

        # --- 4. Format sources ---
        sources = [
            SourceDocument(
                page_content=doc.page_content,
                metadata=doc.metadata,
                score=score,
            )
            for doc, score in docs_with_scores
        ]

        response = QueryResponse(answer=answer, sources=sources)

        # --- 5. Populate cache ---
        _cache[key] = response
        if len(_cache) > _CACHE_MAX_SIZE:
            _cache.popitem(last=False)  # evict oldest

        return response
