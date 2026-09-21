import asyncio
import logging
import time
from typing import Any, Dict, List, Tuple

from langchain_core.documents import Document
from rank_bm25 import BM25Okapi

from database.mongo_db import MongoDBClient

logger = logging.getLogger("api_logger")


class BM25Service:
    def __init__(self, cache_ttl_seconds: int = 300) -> None:
        self._cache_ttl_seconds = cache_ttl_seconds
        self._cache: Dict[str, Any] = {"ts": 0.0}

    async def search(
        self,
        query: str,
        k: int,
        metadata_filter: Dict[str, Any] | None = None,
    ) -> List[Tuple[Document, float]]:
        corpus = await self._get_corpus(metadata_filter)
        docs: List[Document] = corpus.get("documents", [])
        tokenized_docs: List[List[str]] = corpus.get("tokenized_docs", [])
        if not docs:
            return []
        tokenized_query = query.lower().split()
        bm25 = BM25Okapi(tokenized_docs)
        scores = bm25.get_scores(tokenized_query)
        top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k]
        results: List[Tuple[Document, float]] = []
        for idx in top_indices:
            score = float(scores[idx])
            if score <= 0:
                continue
            doc = docs[idx]
            enriched = Document(page_content=doc.page_content, metadata={**doc.metadata, "retrieval_source": "bm25"})
            results.append((enriched, -score))
        results.sort(key=lambda item: item[1])
        return results

    async def _get_corpus(self, metadata_filter: Dict[str, Any] | None) -> Dict[str, Any]:
        cache_key = str(sorted((metadata_filter or {}).items()))
        now = time.time()
        cached_key = self._cache.get("filter_key")
        if cached_key == cache_key and now - float(self._cache.get("ts", 0)) < self._cache_ttl_seconds:
            return self._cache

        db = MongoDBClient.get_db()
        mongo_filter: Dict[str, Any] = {"chunk_type": {"$ne": "parent_document"}}
        if metadata_filter:
            for field, value in metadata_filter.items():
                if value is not None and value != "":
                    mongo_filter[field] = value

        documents: List[Document] = []
        cursor = db["chunks"].find(mongo_filter, {"content": 1, "document_id": 1, "source_filename": 1, "parent_id": 1, "_id": 1})
        async for row in cursor:
            content = row.get("content") or ""
            if not content.strip():
                continue
            documents.append(
                Document(
                    page_content=content,
                    metadata={
                        "chunk_id": str(row.get("_id")),
                        "document_id": row.get("document_id"),
                        "source_filename": row.get("source_filename"),
                        "parent_id": row.get("parent_id"),
                    },
                )
            )

        tokenized_docs = [doc.page_content.lower().split() for doc in documents]
        self._cache = {
            "filter_key": cache_key,
            "ts": now,
            "documents": documents,
            "tokenized_docs": tokenized_docs,
        }
        logger.info("BM25 corpus refreshed documents=%s filter=%s", len(documents), metadata_filter or {})
        return self._cache
