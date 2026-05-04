"""
RagController — handles incoming RAG query requests.

Receives an injected RagService instance (dependency injection)
so the service layer can be easily mocked in tests.
"""
from services.rag_service import RagService
from schemas.rag_schema import QueryRequest, QueryResponse
from fastapi import HTTPException
import logging

logger = logging.getLogger("api_logger")


class RagController:
    """Thin controller layer that delegates to the RagService."""

    def __init__(self, service: RagService | None = None) -> None:
        self.service = service or RagService()

    async def handle_query(self, request: QueryRequest) -> QueryResponse:
        """
        Processes a RAG query and returns a structured response.
        Catches all unexpected exceptions and re-raises as a clean HTTPException.
        """
        try:
            return await self.service.run_pipeline(request)
        except HTTPException:
            raise  # let FastAPI handle HTTP errors directly
        except Exception as exc:
            logger.error("RAG pipeline error: %s", exc, exc_info=True)
            raise HTTPException(
                status_code=500,
                detail="Internal server error during answer generation.",
            )
