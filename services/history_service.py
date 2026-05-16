import logging
from typing import List, Optional
from config.settings import get_settings
from database.mongo_db import MongoDBClient
from helpers.context_helper import trim_history_to_budget
from models.message_model import MessageModel, SourceChunk
logger = logging.getLogger(__name__)
COLLECTION = 'messages'
class HistoryService:
    async def add_message(self, conversation_id: str, role: str, content: str, sources: Optional[List[dict]]=None, tokens_used: int=0) -> MessageModel:
        parsed_sources = [SourceChunk(**s) for s in sources or []]
        msg = MessageModel(conversation_id=conversation_id, role=role, content=content, sources=parsed_sources, tokens_used=tokens_used)
        db = MongoDBClient.get_db()
        await db[COLLECTION].insert_one(msg.to_mongo_dict())
        return msg
    async def get_history(self, conversation_id: str, limit: Optional[int]=None) -> List[MessageModel]:
                   
        settings = get_settings()
        max_turns = limit or settings.MAX_HISTORY_TURNS
        db = MongoDBClient.get_db()
        cursor = db[COLLECTION].find({'conversation_id': conversation_id}).sort('created_at', 1).limit(max_turns * 2)
        docs = await cursor.to_list(length=max_turns * 2)
        history = [MessageModel.from_mongo_dict(d) for d in docs]
        return trim_history_to_budget(history)
    async def get_last_n_messages(self, conversation_id: str, n: int=4) -> List[MessageModel]:
        db = MongoDBClient.get_db()
        cursor = db[COLLECTION].find({'conversation_id': conversation_id}).sort('created_at', -1).limit(n)
        docs = await cursor.to_list(length=n)
        return list(reversed([MessageModel.from_mongo_dict(d) for d in docs]))
    async def clear_history(self, conversation_id: str) -> int:
        db = MongoDBClient.get_db()
        result = await db[COLLECTION].delete_many({'conversation_id': conversation_id})
        return result.deleted_count