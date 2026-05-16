from services.rag_service import RagService
from schemas.rag_schema import QueryRequest, QueryResponse
from fastapi import HTTPException
import logging
logger = logging.getLogger('api_logger')
class RagController:
    def __init__(self, service: RagService | None=None) -> None:
        self.service = service or RagService()
    async def handle_query(self, request: QueryRequest) -> QueryResponse:
        try:
            return await self.service.run_pipeline(request)
        except HTTPException:
            raise
        except Exception as exc:
            logger.error('RAG pipeline error: %s', exc, exc_info=True)
            raise HTTPException(status_code=500, detail='Internal server error during answer generation.')