from typing import List, Optional
import uuid
from database.mongo_db import MongoDBClient
from models.chunk_model import ChunkModel

class ParentDocumentService:
    def __init__(self):
        self.collection = None

    def _get_collection(self):
        if self.collection is None:
            self.collection = MongoDBClient.get_db()["chunks"]
        return self.collection
        
    async def store_parent_document(self, document_id: str, content: str, metadata: dict) -> str:
        parent_id = f"parent_{uuid.uuid4()}"
        parent = ChunkModel(
            chunk_id=parent_id,
            document_id=document_id,
            source_filename=metadata.get("filename", "unknown"),
            content=content,
            chunk_type="parent_document"
        )
        await self._get_collection().insert_one(parent.to_mongo_dict())
        return parent_id
        
    async def get_parent_document(self, parent_id: str) -> Optional[ChunkModel]:
        data = await self._get_collection().find_one({"_id": parent_id})
        if data:
            return ChunkModel.from_mongo_dict(data)
        return None
