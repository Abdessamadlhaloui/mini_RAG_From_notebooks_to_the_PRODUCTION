import logging
from datetime import datetime, timedelta
from typing import List

from database.mongo_db import MongoDBClient
from models.memory_model import MemoryEntry

logger = logging.getLogger("api_logger")


class MemoryService:
    async def get_relevant_memories(self, conversation_id: str, query: str, limit: int) -> List[MemoryEntry]:
        try:
            db = MongoDBClient.get_db()
            terms = [term.lower() for term in query.split() if len(term) > 2]
            cursor = db["memories"].find({"metadata.conversation_id": conversation_id}).sort("access_count", -1).limit(limit * 3)
            rows = await cursor.to_list(length=limit * 3)
            memories = [MemoryEntry.from_mongo_dict(row) for row in rows]
            scored = []
            for memory in memories:
                content = memory.content.lower()
                score = sum(1 for term in terms if term in content)
                scored.append((memory, score))
            scored.sort(key=lambda item: (item[1], item[0].access_count), reverse=True)
            return [memory for memory, _ in scored[:limit]]
        except Exception as exc:
            logger.warning("Memory retrieval failed conversation_id=%s error=%s", conversation_id, exc)
            return []

    async def store_interaction_memory(self, conversation_id: str, query: str, answer: str) -> None:
        try:
            content = f"User asked: {query}\nAssistant answered: {answer[:1000]}"
            memory = MemoryEntry(
                memory_type="conversation",
                content=content,
                metadata={"conversation_id": conversation_id},
                expires_at=datetime.utcnow() + timedelta(days=30),
            )
            db = MongoDBClient.get_db()
            await db["memories"].insert_one(memory.to_mongo_dict())
        except Exception as exc:
            logger.warning("Memory persistence failed conversation_id=%s error=%s", conversation_id, exc)
