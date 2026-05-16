import hashlib
import json
from collections import OrderedDict
from services.retrieval_service import RetrievalService
from services.generation_service import GenerationService
from helpers.text_helper import sanitize_query
from schemas.rag_schema import QueryRequest, QueryResponse, SourceDocument
from config.settings import get_settings
import logging
from fastapi import HTTPException
logger = logging.getLogger('api_logger')
_CACHE_MAX_SIZE = 128
_cache: OrderedDict[str, QueryResponse] = OrderedDict()
def _cache_key(query: str, top_k: int, conversation_id: str='') -> str:
    raw = f'{query.strip().lower()}::{top_k}::{conversation_id}'
    return 'rag_cache:' + hashlib.sha256(raw.encode()).hexdigest()
class RagService:
    def __init__(self, retrieval_service: RetrievalService | None=None, generation_service: GenerationService | None=None) -> None:
        self.retrieval_service = retrieval_service or RetrievalService()
        self.generation_service = generation_service or GenerationService()
        self.settings = get_settings()
    async def run_pipeline(self, request: QueryRequest) -> QueryResponse:
                   
        from database.redis_db import RedisClient
        from helpers.context_helper import format_history_for_prompt
        from services.conversation_service import ConversationService
        from services.history_service import HistoryService
        top_k = request.top_k or self.settings.top_k
        clean_query = sanitize_query(request.query)
        conv_service = ConversationService()
        hist_service = HistoryService()
        try:
            conversation = await conv_service.get_or_create(request.conversation_id)
        except ValueError as e:
            raise HTTPException(status_code=404, detail=str(e))
        conv_id = conversation.conversation_id
        history = await hist_service.get_history(conv_id)
        history_for_prompt = format_history_for_prompt(history)
        key = _cache_key(clean_query, top_k, conv_id)
        cached_json = await RedisClient.get_cache(key)
        if cached_json:
            logger.info('Redis cache HIT for conv %s key %s', conv_id, key[:12])
            payload = json.loads(cached_json)
            sources = [SourceDocument(page_content=s.get('page_content') or s.get('content') or '', metadata=s.get('metadata') or {}, score=s.get('score')) for s in payload.get('sources', [])]
            return QueryResponse(answer=payload.get('answer', ''), sources=sources, conversation_id=conv_id, tokens_used=payload.get('tokens_used'), cached=True)
        if key in _cache:
            logger.info('Local cache HIT for conv %s key %s', conv_id, key[:12])
            cached = _cache[key]
            return QueryResponse(answer=cached.answer, sources=cached.sources, conversation_id=conv_id, tokens_used=cached.tokens_used, cached=True)
        docs_with_scores = await self.retrieval_service.retrieve_context(clean_query, top_k)
        if not docs_with_scores:
            return QueryResponse(answer="I couldn't find any relevant context in the database to answer your question.", sources=[], conversation_id=conv_id)
        context_chunks = [doc.page_content for doc, _ in docs_with_scores]
        answer, tokens_used = await self.generation_service.generate_answer_with_history(query=clean_query, context_chunks=context_chunks, history_messages=history_for_prompt)
        await hist_service.add_message(conv_id, 'user', clean_query)
        await hist_service.add_message(conv_id, 'assistant', answer, sources=[{'content': doc.page_content, 'score': float(score)} for doc, score in docs_with_scores], tokens_used=tokens_used)
        if conversation.message_count == 0:
            auto_title = clean_query[:self.settings.CONVERSATION_TITLE_MAX_CHARS].rstrip()
            await conv_service.update_title(conv_id, auto_title)
        await conv_service.increment_message_count(conv_id)
        await conv_service.increment_message_count(conv_id)
        sources = [SourceDocument(page_content=doc.page_content, metadata=doc.metadata, score=score) for doc, score in docs_with_scores]
        response = QueryResponse(answer=answer, sources=sources, conversation_id=conv_id, tokens_used=tokens_used)
        _cache[key] = response
        if len(_cache) > _CACHE_MAX_SIZE:
            _cache.popitem(last=False)
        try:
            await RedisClient.set_cache(key, json.dumps({'answer': response.answer, 'sources': [s.model_dump() for s in response.sources], 'conversation_id': conv_id, 'tokens_used': tokens_used}))
        except Exception:
            pass
        return response