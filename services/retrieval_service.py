import asyncio
import logging
from typing import Any, Dict, List, Optional, Tuple

from langchain_core.documents import Document

from config.settings import get_settings
from services.bm25_service import BM25Service
from services.embedding_service import EmbeddingService
from services.query_enhancement_service import GraphRetrievalService, QueryEnhancementService
from services.parent_context_service import ParentContextService
from services.reranking_service import RerankingService
from services.vector_db_service import VectorDBService

logger = logging.getLogger("api_logger")


class RetrievalService:
    def __init__(
        self,
        embedding_service: EmbeddingService | None = None,
        vector_db_service: VectorDBService | None = None,
        bm25_service: BM25Service | None = None,
        query_enhancement_service: QueryEnhancementService | None = None,
        graph_retrieval_service: GraphRetrievalService | None = None,
        reranking_service: RerankingService | None = None,
        parent_context_service: ParentContextService | None = None,
    ) -> None:
        settings = get_settings()
        self._settings = settings
        self.embedding_service = embedding_service or EmbeddingService()
        self.vector_db = vector_db_service or VectorDBService()
        self.bm25_service = bm25_service or BM25Service(cache_ttl_seconds=settings.RETRIEVAL_BM25_CACHE_TTL_SECONDS)
        self.query_enhancement_service = query_enhancement_service or QueryEnhancementService()
        self.graph_retrieval_service = graph_retrieval_service or GraphRetrievalService()
        self.reranking_service = reranking_service or RerankingService()
        self.parent_context_service = parent_context_service or ParentContextService()

    async def retrieve_context(
        self,
        query: str,
        top_k: int,
        metadata_filter: Optional[Dict[str, Any]] = None,
    ) -> List[Tuple[Document, float]]:
        semantic_query, inferred_filter = await self.query_enhancement_service.parse_self_query(query)
        merged_filter = self._merge_filters(metadata_filter, inferred_filter)
        query_variants = await self.query_enhancement_service.build_query_variants(semantic_query)
        hyde_document = await self.query_enhancement_service.generate_hyde_document(semantic_query)
        if hyde_document:
            query_variants.append(hyde_document)

        initial_k = max(top_k, self._settings.RETRIEVAL_INITIAL_K)
        dense_tasks = [self._semantic_search(text, initial_k, merged_filter) for text in query_variants]
        dense_results_nested = await asyncio.gather(*dense_tasks, return_exceptions=True)
        dense_results: List[List[Tuple[Document, float]]] = []
        for result in dense_results_nested:
            if isinstance(result, Exception):
                logger.warning("Dense retrieval failed: %s", result)
                continue
            dense_results.append(result)

        result_lists: List[List[Tuple[Document, float]]] = list(dense_results)
        if self._settings.RETRIEVAL_ENABLE_BM25:
            bm25_results = await self.bm25_service.search(semantic_query, initial_k, merged_filter)
            if bm25_results:
                result_lists.append(bm25_results)
        if self._settings.RETRIEVAL_ENABLE_GRAPH:
            graph_results = await self.graph_retrieval_service.search(semantic_query, initial_k, merged_filter)
            if graph_results:
                result_lists.append(graph_results)

        if not result_lists:
            return []

        merged = self._reciprocal_rank_fusion(result_lists, k=self._settings.RETRIEVAL_RRF_K)
        rerank_k = max(top_k, self._settings.RETRIEVAL_FINAL_K)
        reranked = await self.reranking_service.rerank(semantic_query, merged, rerank_k)
        with_parent_context = await self.parent_context_service.expand(reranked)
        return with_parent_context[:top_k]

    async def _semantic_search(
        self,
        query: str,
        k: int,
        metadata_filter: Optional[Dict[str, Any]],
    ) -> List[Tuple[Document, float]]:
        query_embedding = await self.embedding_service.get_query_embedding(query)
        docs_with_scores = await self.vector_db.search(query_embedding, top_k=k, filter_criteria=metadata_filter)
        normalized: List[Tuple[Document, float]] = []
        for doc, score in docs_with_scores:
            metadata = dict(doc.metadata)
            metadata["retrieval_source"] = metadata.get("retrieval_source") or "dense"
            normalized.append((Document(page_content=doc.page_content, metadata=metadata), score))
        normalized.sort(key=lambda item: item[1])
        return normalized

    @staticmethod
    def _merge_filters(
        explicit: Optional[Dict[str, Any]],
        inferred: Optional[Dict[str, Any]],
    ) -> Optional[Dict[str, Any]]:
        merged: Dict[str, Any] = {}
        for source in (inferred or {}, explicit or {}):
            for key, value in source.items():
                if value is not None and value != "":
                    merged[key] = value
        return merged or None

    @staticmethod
    def _document_key(doc: Document) -> str:
        chunk_id = doc.metadata.get("chunk_id")
        if chunk_id:
            return str(chunk_id)
        parent_chunk = doc.metadata.get("parent_chunk_id")
        vector_type = doc.metadata.get("vector_type")
        if parent_chunk and vector_type:
            return f"{parent_chunk}:{vector_type}:{doc.page_content[:80]}"
        return doc.page_content[:160]

    @classmethod
    def _reciprocal_rank_fusion(
        result_lists: List[List[Tuple[Document, float]]],
        k: int = 60,
    ) -> List[Tuple[Document, float]]:
        scores: Dict[str, float] = {}
        doc_map: Dict[str, Document] = {}
        for result_list in result_lists:
            for rank, (doc, _score) in enumerate(result_list):
                key = RetrievalService._document_key(doc)
                scores[key] = scores.get(key, 0.0) + 1.0 / (k + rank + 1)
                doc_map[key] = doc
        sorted_keys = sorted(scores, key=lambda item: scores[item], reverse=True)
        merged: List[Tuple[Document, float]] = [
            (doc_map[key], -round(scores[key], 6)) for key in sorted_keys
        ]
        merged.sort(key=lambda item: item[1])
        return merged
