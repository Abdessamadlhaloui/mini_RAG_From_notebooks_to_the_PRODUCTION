import logging
from datetime import datetime
from typing import List, Optional, Tuple
from pymongo import ReturnDocument
from config.settings import get_settings
from database.mongo_db import MongoDBClient
from models.conversation_model import ConversationModel
logger = logging.getLogger(__name__)
COLLECTION = 'conversations'
class ConversationService:
    async def create_conversation(self, title: Optional[str]=None) -> ConversationModel:
        settings = get_settings()
        conv = ConversationModel(title=(title or 'New conversation')[:settings.CONVERSATION_TITLE_MAX_CHARS])
        db = MongoDBClient.get_db()
        await db[COLLECTION].insert_one(conv.to_mongo_dict())
        logger.info(f'Created conversation {conv.conversation_id}')
        return conv
    async def get_conversation(self, conversation_id: str) -> Optional[ConversationModel]:
        db = MongoDBClient.get_db()
        doc = await db[COLLECTION].find_one({'_id': conversation_id})
        if not doc:
            return None
        return ConversationModel.from_mongo_dict(doc)
    async def get_or_create(self, conversation_id: Optional[str]=None) -> ConversationModel:
                   
        if conversation_id:
            conv = await self.get_conversation(conversation_id)
            if not conv:
                raise ValueError(f"Conversation '{conversation_id}' not found.")
            return conv
        return await self.create_conversation()
    async def list_conversations(self, page: int=1, limit: int=20) -> Tuple[List[ConversationModel], int]:
        db = MongoDBClient.get_db()
        skip = (page - 1) * limit
        total = await db[COLLECTION].count_documents({})
        cursor = db[COLLECTION].find({}).sort('updated_at', -1).skip(skip).limit(limit)
        docs = await cursor.to_list(length=limit)
        return ([ConversationModel.from_mongo_dict(d) for d in docs], total)
    async def update_title(self, conversation_id: str, title: str) -> Optional[ConversationModel]:
        settings = get_settings()
        db = MongoDBClient.get_db()
        now = datetime.utcnow()
        result = await db[COLLECTION].find_one_and_update({'_id': conversation_id}, {'$set': {'title': title[:settings.CONVERSATION_TITLE_MAX_CHARS], 'updated_at': now}}, return_document=ReturnDocument.AFTER)
        if not result:
            return None
        return ConversationModel.from_mongo_dict(result)
    async def increment_message_count(self, conversation_id: str) -> None:
        db = MongoDBClient.get_db()
        await db[COLLECTION].update_one({'_id': conversation_id}, {'$inc': {'message_count': 1}, '$set': {'updated_at': datetime.utcnow()}})
    async def delete_conversation(self, conversation_id: str) -> bool:
                                                        
        db = MongoDBClient.get_db()
        conv_result = await db[COLLECTION].delete_one({'_id': conversation_id})
        await db['messages'].delete_many({'conversation_id': conversation_id})
        deleted = conv_result.deleted_count > 0
        if deleted:
            logger.info(f'Deleted conversation {conversation_id} and its messages.')
        return deleted