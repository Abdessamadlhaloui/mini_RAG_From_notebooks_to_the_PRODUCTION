from services.embedding_service import EmbeddingService
from services.vector_db_service import VectorDBService
from langchain.docstore.document import Document
import asyncio
import logging
import time
from typing import Dict, List, Tuple
logger = logging.getLogger('api_logger')
class RetrievalService:
    def __init__(self, embedding_service: EmbeddingService | None=None, vector_db_service: VectorDBService | None=None) -> None:
        self.embedding_service = embedding_service or EmbeddingService()
        self.vector_db = vector_db_service or VectorDBService()
    async def retrieve_context(self, query: str, top_k: int) -> List[Tuple[Document, float]]:
                   
        semantic_results = await self._semantic_search(query, k=top_k * 2)
        bm25_results = await self._bm25_search(query, k=top_k * 2)
        if not bm25_results:
            return semantic_results[:top_k]
        merged = self._reciprocal_rank_fusion([semantic_results, bm25_results])
        return merged[:top_k]
    async def _semantic_search(self, query: str, k: int) -> List[Tuple[Document, float]]:
        query_embedding = await self.embedding_service.get_query_embedding(query)
        docs_with_scores = await self.vector_db.search(query_embedding, top_k=k)
        docs_with_scores.sort(key=lambda x: x[1])
        return docs_with_scores
    async def _bm25_search(self, query: str, k: int) -> List[Tuple[Document, float]]:
                   
        from rank_bm25 import BM25Okapi
        corpus = await self._get_bm25_corpus()
        docs = corpus.get('docs', [])
        tokenized_docs = corpus.get('tokenized_docs', [])
        if not docs:
            return []
        tokenized_query = query.lower().split()
        bm25 = BM25Okapi(tokenized_docs)
        scores = bm25.get_scores(tokenized_query)
        top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k]
        results: List[Tuple[Document, float]] = []
        for idx in top_indices:
            s = float(scores[idx])
            if s > 0:
                doc = Document(page_content=docs[idx], metadata={'source': 'bm25'})
                results.append((doc, -s))
        results.sort(key=lambda x: x[1])
        return results
    async def _get_bm25_corpus(self) -> Dict[str, object]:
                   
        if not hasattr(self, '_corpus_cache') or time.time() - self._corpus_cache.get('ts', 0) > 300:
            loop = asyncio.get_running_loop()
            def _load_all_docs() -> List[str]:
                data = self.vector_db.collection.get(include=['documents'])
                return list(data.get('documents') or [])
            try:
                docs: List[str] = await loop.run_in_executor(None, _load_all_docs)
            except Exception as exc:
                logger.warning('BM25 corpus load failed (non-fatal): %s', exc)
                docs = []
            self._corpus_cache = {'docs': docs, 'tokenized_docs': [d.lower().split() for d in docs], 'ts': time.time()}
        return self._corpus_cache
    @staticmethod
    def _reciprocal_rank_fusion(result_lists: List[List[Tuple[Document, float]]], k: int=60) -> List[Tuple[Document, float]]:
        scores: Dict[str, float] = {}
        doc_map: Dict[str, Document] = {}
        for result_list in result_lists:
            for rank, (doc, _score) in enumerate(result_list):
                key = doc.page_content[:120]
                scores[key] = scores.get(key, 0.0) + 1.0 / (k + rank + 1)
                doc_map[key] = doc
        sorted_keys = sorted(scores, key=lambda x: scores[x], reverse=True)
        merged: List[Tuple[Document, float]] = [(doc_map[key], -round(scores[key], 6)) for key in sorted_keys]
        merged.sort(key=lambda x: x[1])
        return merged