import logging
from fastapi import HTTPException
from schemas.conversation_schema import ConversationDetailResponse, ConversationListResponse, ConversationSummaryResponse, CreateConversationRequest, MessageResponse, UpdateConversationRequest
from services.conversation_service import ConversationService
from services.history_service import HistoryService
logger = logging.getLogger(__name__)
class ConversationController:
    def __init__(self) -> None:
        self.conv_service = ConversationService()
        self.hist_service = HistoryService()
    async def create(self, request: CreateConversationRequest) -> ConversationSummaryResponse:
        conv = await self.conv_service.create_conversation(title=request.title)
        return ConversationSummaryResponse(conversation_id=conv.conversation_id, title=conv.title, message_count=conv.message_count, created_at=conv.created_at, updated_at=conv.updated_at)
    async def list_all(self, page: int, limit: int) -> ConversationListResponse:
        convs, total = await self.conv_service.list_conversations(page=page, limit=limit)
        return ConversationListResponse(conversations=[ConversationSummaryResponse(conversation_id=c.conversation_id, title=c.title, message_count=c.message_count, created_at=c.created_at, updated_at=c.updated_at) for c in convs], total=total, page=page, limit=limit)
    async def get_detail(self, conversation_id: str) -> ConversationDetailResponse:
        conv = await self.conv_service.get_conversation(conversation_id)
        if not conv:
            raise HTTPException(status_code=404, detail='Conversation not found.')
        history = await self.hist_service.get_history(conversation_id, limit=100)
        messages = [MessageResponse(message_id=m.message_id, role=m.role, content=m.content, sources=[s.model_dump() for s in m.sources], tokens_used=m.tokens_used, created_at=m.created_at) for m in history]
        return ConversationDetailResponse(conversation_id=conv.conversation_id, title=conv.title, message_count=conv.message_count, created_at=conv.created_at, updated_at=conv.updated_at, messages=messages)
    async def update(self, conversation_id: str, request: UpdateConversationRequest) -> ConversationSummaryResponse:
        conv = await self.conv_service.update_title(conversation_id, request.title)
        if not conv:
            raise HTTPException(status_code=404, detail='Conversation not found.')
        return ConversationSummaryResponse(conversation_id=conv.conversation_id, title=conv.title, message_count=conv.message_count, created_at=conv.created_at, updated_at=conv.updated_at)
    async def delete(self, conversation_id: str) -> dict:
        deleted = await self.conv_service.delete_conversation(conversation_id)
        if not deleted:
            raise HTTPException(status_code=404, detail='Conversation not found.')
        return {'deleted': True, 'conversation_id': conversation_id}