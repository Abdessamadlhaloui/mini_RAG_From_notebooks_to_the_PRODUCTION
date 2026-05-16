from fastapi import APIRouter, Depends, Query
from controllers.conversation_controller import ConversationController
from middlewares.auth_middleware import verify_api_key
from schemas.conversation_schema import CreateConversationRequest, UpdateConversationRequest
router = APIRouter(prefix='/api/v1/conversations', tags=['Conversations'])
def get_controller() -> ConversationController:
    return ConversationController()
@router.post('', dependencies=[Depends(verify_api_key)])
async def create_conversation(request: CreateConversationRequest, controller: ConversationController=Depends(get_controller)):
    return await controller.create(request)
@router.get('', dependencies=[Depends(verify_api_key)])
async def list_conversations(page: int=Query(default=1, ge=1), limit: int=Query(default=20, ge=1, le=100), controller: ConversationController=Depends(get_controller)):
    return await controller.list_all(page=page, limit=limit)
@router.get('/{conversation_id}', dependencies=[Depends(verify_api_key)])
async def get_conversation(conversation_id: str, controller: ConversationController=Depends(get_controller)):
    return await controller.get_detail(conversation_id)
@router.patch('/{conversation_id}', dependencies=[Depends(verify_api_key)])
async def update_conversation(conversation_id: str, request: UpdateConversationRequest, controller: ConversationController=Depends(get_controller)):
    return await controller.update(conversation_id, request)
@router.delete('/{conversation_id}', dependencies=[Depends(verify_api_key)])
async def delete_conversation(conversation_id: str, controller: ConversationController=Depends(get_controller)):
    return await controller.delete(conversation_id)