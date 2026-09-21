import asyncio
import json
import logging
import re
from typing import Any, Dict, List, Tuple

from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

from config.settings import get_settings
from database.neo4j_db import Neo4jClient

logger = logging.getLogger("api_logger")


class GraphRetrievalService:
    async def index_chunk(self, chunk_id: str, document_id: str, content: str, metadata: Dict[str, Any]) -> None:
        driver = Neo4jClient.get_driver()
        if driver is None:
            return
        query = """
        MERGE (d:Document {document_id: $document_id})
        MERGE (c:Chunk {chunk_id: $chunk_id})
        SET c.content = $content,
            c.source_filename = $source_filename,
            c.parent_id = $parent_id,
            c.updated_at = timestamp()
        MERGE (d)-[:HAS_CHUNK]->(c)
        """
        params = {
            "chunk_id": chunk_id,
            "document_id": document_id,
            "content": content,
            "source_filename": metadata.get("source_filename"),
            "parent_id": metadata.get("parent_id"),
        }
        try:
            await Neo4jClient.execute_query(query, params)
        except Exception as exc:
            logger.warning("Neo4j chunk index failed chunk_id=%s error=%s", chunk_id, exc)

    async def search(self, query: str, k: int, metadata_filter: Dict[str, Any] | None = None) -> List[Tuple[Document, float]]:
        driver = Neo4jClient.get_driver()
        if driver is None:
            return []
        terms = [t for t in re.split(r"\W+", query.lower()) if len(t) > 2][:8]
        if not terms:
            return []
        document_id = (metadata_filter or {}).get("document_id")
        cypher = """
        MATCH (c:Chunk)
        WHERE ANY(term IN $terms WHERE toLower(c.content) CONTAINS term)
        OPTIONAL MATCH (d:Document)-[:HAS_CHUNK]->(c)
        WITH c, d, size([term IN $terms WHERE toLower(c.content) CONTAINS term]) AS match_count
        WHERE $document_id IS NULL OR d.document_id = $document_id
        RETURN c.content AS content,
               c.chunk_id AS chunk_id,
               c.source_filename AS source_filename,
               c.parent_id AS parent_id,
               match_count AS match_count
        ORDER BY match_count DESC
        LIMIT $k
        """
        try:
            rows = await Neo4jClient.execute_query(
                cypher,
                {"terms": terms, "k": k, "document_id": document_id},
            )
        except Exception as exc:
            logger.warning("Graph retrieval failed: %s", exc)
            return []
        results: List[Tuple[Document, float]] = []
        for row in rows:
            content = row.get("content") or ""
            if not content:
                continue
            metadata = {
                "chunk_id": row.get("chunk_id"),
                "source_filename": row.get("source_filename"),
                "parent_id": row.get("parent_id"),
                "retrieval_source": "graph",
            }
            if document_id:
                metadata["document_id"] = document_id
            match_count = float(row.get("match_count") or 0)
            results.append((Document(page_content=content, metadata=metadata), -match_count))
        results.sort(key=lambda item: item[1])
        return results


class QueryEnhancementService:
    def __init__(self) -> None:
        settings = get_settings()
        self._settings = settings
        self._llm = None
        if settings.openai_api_key:
            try:
                self._llm = ChatOpenAI(
                    model="gpt-4o-mini",
                    temperature=0,
                    openai_api_key=settings.openai_api_key,
                    timeout=30.0,
                )
            except Exception as exc:
                logger.warning("Query enhancement LLM disabled: %s", exc)
        self._multi_query_prompt = ChatPromptTemplate.from_template(
            "Generate {count} diverse search queries that help retrieve documents answering the user question.\n"
            "Return one query per line without numbering.\n\nQuestion: {question}"
        )
        self._expansion_prompt = ChatPromptTemplate.from_template(
            "Expand this search query with closely related keywords and phrases on one line.\n\nQuery: {question}"
        )
        self._hyde_prompt = ChatPromptTemplate.from_template(
            "Write a concise factual paragraph that would answer this question if the knowledge existed in a document.\n\nQuestion: {question}"
        )
        self._self_query_prompt = ChatPromptTemplate.from_template(
            'Extract metadata filters and a semantic query from the user question.\n'
            'Return strict JSON with keys: semantic_query (string), document_id (string or null), source_filename (string or null).\n\n'
            "Question: {question}"
        )

    async def build_query_variants(self, query: str) -> List[str]:
        variants = [query.strip()]
        settings = self._settings
        tasks = []
        if settings.RETRIEVAL_ENABLE_MULTI_QUERY:
            tasks.append(self._generate_multi_queries(query))
        if settings.RETRIEVAL_ENABLE_QUERY_EXPANSION:
            tasks.append(self._expand_query(query))
        if tasks:
            results = await asyncio.gather(*tasks, return_exceptions=True)
            for result in results:
                if isinstance(result, Exception):
                    logger.warning("Query enhancement step failed: %s", result)
                    continue
                for item in result:
                    if item and item not in variants:
                        variants.append(item)
        return variants

    async def generate_hyde_document(self, query: str) -> str:
        if not self._settings.RETRIEVAL_ENABLE_HYDE:
            return ""
        if self._llm is None:
            return self._fallback_hyde(query)
        try:
            chain = self._hyde_prompt | self._llm
            response = await chain.ainvoke({"question": query})
            return (response.content or "").strip()
        except Exception as exc:
            logger.warning("HyDE generation failed: %s", exc)
            return self._fallback_hyde(query)

    async def parse_self_query(self, query: str) -> Tuple[str, Dict[str, Any]]:
        if not self._settings.RETRIEVAL_ENABLE_SELF_QUERY:
            return query, {}
        if self._llm is None:
            return query, self._fallback_filter(query)
        try:
            chain = self._self_query_prompt | self._llm
            response = await chain.ainvoke({"question": query})
            payload = json.loads(response.content.strip())
            semantic_query = str(payload.get("semantic_query") or query).strip()
            metadata_filter: Dict[str, Any] = {}
            document_id = payload.get("document_id")
            source_filename = payload.get("source_filename")
            if document_id:
                metadata_filter["document_id"] = document_id
            if source_filename:
                metadata_filter["source_filename"] = source_filename
            return semantic_query or query, metadata_filter
        except Exception as exc:
            logger.warning("Self-query parsing failed: %s", exc)
            return query, self._fallback_filter(query)

    async def _generate_multi_queries(self, query: str) -> List[str]:
        if self._llm is None:
            return self._fallback_multi_queries(query)
        chain = self._multi_query_prompt | self._llm
        response = await chain.ainvoke({"question": query, "count": self._settings.RETRIEVAL_MULTI_QUERY_COUNT})
        lines = [line.strip("- ").strip() for line in (response.content or "").splitlines() if line.strip()]
        return lines[: self._settings.RETRIEVAL_MULTI_QUERY_COUNT] or self._fallback_multi_queries(query)

    async def _expand_query(self, query: str) -> List[str]:
        if self._llm is None:
            return self._fallback_expansion(query)
        chain = self._expansion_prompt | self._llm
        response = await chain.ainvoke({"question": query})
        expanded = (response.content or "").strip()
        return [expanded] if expanded else self._fallback_expansion(query)

    @staticmethod
    def _fallback_multi_queries(query: str) -> List[str]:
        tokens = [token for token in re.split(r"\W+", query) if len(token) > 3]
        variants = [query.strip()]
        for token in tokens[:2]:
            variants.append(query.replace(token, f"{token} related").strip())
        return variants[:3]

    @staticmethod
    def _fallback_expansion(query: str) -> List[str]:
        return [f"{query.strip()} details", f"{query.strip()} explanation"]

    @staticmethod
    def _fallback_hyde(query: str) -> str:
        return f"This document likely discusses {query.strip()} in a factual, technical context."

    @staticmethod
    def _fallback_filter(query: str) -> Tuple[str, Dict[str, Any]]:
        lowered = query.lower()
        metadata_filter: Dict[str, Any] = {}
        match = re.search(r"document\s+([a-z0-9_\-]+)", lowered)
        if match:
            metadata_filter["document_id"] = match.group(1)
        match = re.search(r"file\s+([a-z0-9_\-]+\.[a-z0-9]+)", lowered)
        if match:
            metadata_filter["source_filename"] = match.group(1)
        return query, metadata_filter
