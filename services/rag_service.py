import hashlib
import json
import logging
from collections import OrderedDict
from typing import AsyncIterator

from fastapi import HTTPException

from config.settings import get_settings
from helpers.text_helper import sanitize_query
from schemas.rag_schema import QueryRequest, QueryResponse, SourceDocument
from services.evaluation_service import EvaluationService, LatencyTimer
from services.generation_service import GenerationService
from services.memory_service import MemoryService
from services.observability_service import CostTrackingService, MetricsService
from services.rag_quality_service import RAGQualityService, RetrievalExplainabilityService
from services.retrieval_service import RetrievalService

logger = logging.getLogger("api_logger")

_CACHE_MAX_SIZE = 128
_cache: OrderedDict[str, QueryResponse] = OrderedDict()


def _cache_key(query: str, top_k: int, conversation_id: str = "") -> str:
    raw = f"{query.strip().lower()}::{top_k}::{conversation_id}"
    return "rag_cache:" + hashlib.sha256(raw.encode()).hexdigest()


class RagService:
    def __init__(
        self,
        retrieval_service: RetrievalService | None = None,
        generation_service: GenerationService | None = None,
        quality_service: RAGQualityService | None = None,
        evaluation_service: EvaluationService | None = None,
        memory_service: MemoryService | None = None,
    ) -> None:
        self.retrieval_service = retrieval_service or RetrievalService()
        self.generation_service = generation_service or GenerationService()
        self.quality_service = quality_service or RAGQualityService()
        self.evaluation_service = evaluation_service or EvaluationService()
        self.memory_service = memory_service or MemoryService()
        self.explainability_service = RetrievalExplainabilityService()
        self.cost_service = CostTrackingService()
        self.settings = get_settings()

    async def run_pipeline(self, request: QueryRequest) -> QueryResponse:
        from database.redis_db import RedisClient
        from helpers.context_helper import format_history_for_prompt
        from services.conversation_service import ConversationService
        from services.history_service import HistoryService

        timer = LatencyTimer()
        MetricsService.increment("query_total")
        top_k = request.top_k or self.settings.top_k
        clean_query = sanitize_query(request.query)
        conv_service = ConversationService()
        hist_service = HistoryService()
        try:
            conversation = await conv_service.get_or_create(request.conversation_id)
        except ValueError as exc:
            MetricsService.increment("query_error_total")
            raise HTTPException(status_code=404, detail=str(exc)) from exc

        conv_id = conversation.conversation_id
        history = await hist_service.get_history(conv_id)
        history_for_prompt = format_history_for_prompt(history)
        key = _cache_key(clean_query, top_k, conv_id)
        cached = await self._get_cached_response(key, conv_id, RedisClient)
        if cached:
            MetricsService.increment("query_cache_hit_total")
            return cached

        docs_with_scores = await self.retrieval_service.retrieve_context(
            clean_query,
            top_k,
            metadata_filter=request.metadata_filter,
        )
        if not docs_with_scores:
            MetricsService.increment("query_empty_retrieval_total")
            return QueryResponse(
                answer="I couldn't find any relevant context in the database to answer your question.",
                sources=[],
                conversation_id=conv_id,
                confidence_score=0.0,
            )

        memory_entries = await self.memory_service.get_relevant_memories(
            conv_id,
            clean_query,
            self.settings.MEMORY_MAX_RELEVANT_ITEMS,
        )
        memory_items = [memory.content for memory in memory_entries]
        context_chunks = [doc.page_content for doc, _ in docs_with_scores]
        answer, tokens_used = await self.generation_service.generate_answer_with_history(
            query=clean_query,
            context_chunks=context_chunks,
            history_messages=history_for_prompt,
            memory_items=memory_items,
        )
        citations = self.quality_service.build_citations(docs_with_scores)
        if self.settings.GENERATION_ENABLE_VERIFICATION:
            verification = self.quality_service.verify_answer(clean_query, answer, docs_with_scores)
        else:
            verification = None
        if citations:
            answer = self.quality_service.append_inline_citations(answer, citations)
        confidence = self.quality_service.confidence_score(verification, docs_with_scores) if verification else None
        cost = self.cost_service.estimate(tokens_used, answer)

        await hist_service.add_message(conv_id, "user", clean_query)
        await hist_service.add_message(
            conv_id,
            "assistant",
            answer,
            sources=[{"content": doc.page_content, "score": float(score)} for doc, score in docs_with_scores],
            tokens_used=tokens_used,
        )
        await self.memory_service.store_interaction_memory(conv_id, clean_query, answer)
        if conversation.message_count == 0:
            auto_title = clean_query[: self.settings.CONVERSATION_TITLE_MAX_CHARS].rstrip()
            await conv_service.update_title(conv_id, auto_title)
        await conv_service.increment_message_count(conv_id)
        await conv_service.increment_message_count(conv_id)

        sources = [SourceDocument(page_content=doc.page_content, metadata=doc.metadata, score=score) for doc, score in docs_with_scores]
        evaluation_id = None
        if self.settings.EVALUATION_ENABLE_BACKGROUND:
            evaluation_id = await self.evaluation_service.record_query_evaluation(
                clean_query,
                answer,
                docs_with_scores,
                verification,
                timer.elapsed_ms,
            )
        response = QueryResponse(
            answer=answer,
            sources=sources,
            conversation_id=conv_id,
            tokens_used=tokens_used,
            cached=False,
            citations=citations,
            verification=verification,
            confidence_score=confidence,
            estimated_cost_usd=cost.estimated_cost_usd,
            evaluation_id=evaluation_id,
        )
        _cache[key] = response
        if len(_cache) > _CACHE_MAX_SIZE:
            _cache.popitem(last=False)
        await self._set_cached_response(key, response, RedisClient)
        MetricsService.observe_latency("query_latency", timer.elapsed_ms)
        MetricsService.increment("tokens_total", tokens_used or 0)
        MetricsService.increment("estimated_cost_usd_total", cost.estimated_cost_usd)
        MetricsService.log_event(
            "rag_query_completed",
            conversation_id=conv_id,
            top_k=top_k,
            latency_ms=timer.elapsed_ms,
            confidence_score=confidence,
            evaluation_id=evaluation_id,
            retrieval=self.explainability_service.explain(docs_with_scores),
        )
        return response

    async def stream_pipeline(self, request: QueryRequest) -> AsyncIterator[str]:
        from helpers.context_helper import format_history_for_prompt
        from services.conversation_service import ConversationService
        from services.history_service import HistoryService

        top_k = request.top_k or self.settings.top_k
        clean_query = sanitize_query(request.query)
        conv_service = ConversationService()
        hist_service = HistoryService()
        try:
            conversation = await conv_service.get_or_create(request.conversation_id)
        except ValueError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        conv_id = conversation.conversation_id
        history = await hist_service.get_history(conv_id)
        docs_with_scores = await self.retrieval_service.retrieve_context(
            clean_query,
            top_k,
            metadata_filter=request.metadata_filter,
        )
        citations = self.quality_service.build_citations(docs_with_scores)
        yield self._json_line({"type": "metadata", "conversation_id": conv_id, "citations": [c.model_dump() for c in citations]})
        if not docs_with_scores:
            answer = "I couldn't find any relevant context in the database to answer your question."
            yield self._json_line({"type": "token", "content": answer})
            yield self._json_line({"type": "done", "conversation_id": conv_id})
            return
        memory_entries = await self.memory_service.get_relevant_memories(
            conv_id,
            clean_query,
            self.settings.MEMORY_MAX_RELEVANT_ITEMS,
        )
        content_parts: list[str] = []
        async for token in self.generation_service.stream_answer_with_history(
            query=clean_query,
            context_chunks=[doc.page_content for doc, _ in docs_with_scores],
            history_messages=format_history_for_prompt(history),
            memory_items=[memory.content for memory in memory_entries],
        ):
            content_parts.append(token)
            yield self._json_line({"type": "token", "content": token})
        answer = "".join(content_parts)
        verification = self.quality_service.verify_answer(clean_query, answer, docs_with_scores)
        confidence = self.quality_service.confidence_score(verification, docs_with_scores)
        await hist_service.add_message(conv_id, "user", clean_query)
        await hist_service.add_message(
            conv_id,
            "assistant",
            answer,
            sources=[{"content": doc.page_content, "score": float(score)} for doc, score in docs_with_scores],
            tokens_used=0,
        )
        await self.memory_service.store_interaction_memory(conv_id, clean_query, answer)
        yield self._json_line(
            {
                "type": "done",
                "conversation_id": conv_id,
                "verification": verification.model_dump(),
                "confidence_score": confidence,
            }
        )

    async def _get_cached_response(self, key: str, conv_id: str, redis_client) -> QueryResponse | None:
        cached_json = await redis_client.get_cache(key)
        if cached_json:
            payload = json.loads(cached_json)
            sources = [
                SourceDocument(
                    page_content=s.get("page_content") or s.get("content") or "",
                    metadata=s.get("metadata") or {},
                    score=s.get("score"),
                )
                for s in payload.get("sources", [])
            ]
            return QueryResponse(
                answer=payload.get("answer", ""),
                sources=sources,
                conversation_id=conv_id,
                tokens_used=payload.get("tokens_used"),
                cached=True,
                citations=payload.get("citations", []),
                verification=payload.get("verification"),
                confidence_score=payload.get("confidence_score"),
                estimated_cost_usd=payload.get("estimated_cost_usd"),
                evaluation_id=payload.get("evaluation_id"),
            )
        if key in _cache:
            cached = _cache[key]
            return QueryResponse(
                answer=cached.answer,
                sources=cached.sources,
                conversation_id=conv_id,
                tokens_used=cached.tokens_used,
                cached=True,
                citations=cached.citations,
                verification=cached.verification,
                confidence_score=cached.confidence_score,
                estimated_cost_usd=cached.estimated_cost_usd,
                evaluation_id=cached.evaluation_id,
            )
        return None

    @staticmethod
    async def _set_cached_response(key: str, response: QueryResponse, redis_client) -> None:
        try:
            await redis_client.set_cache(key, response.model_dump_json())
        except Exception:
            return

    @staticmethod
    def _json_line(payload: dict) -> str:
        return json.dumps(payload, default=str) + "\n"
