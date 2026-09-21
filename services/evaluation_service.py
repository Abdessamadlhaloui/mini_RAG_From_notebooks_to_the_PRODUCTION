import logging
import time
from typing import Any, Dict, List, Tuple

from langchain_core.documents import Document

from database.mongo_db import MongoDBClient
from models.evaluation_model import EvaluationResult
from schemas.rag_schema import VerificationResult
from services.ragas_service import RAGASService

logger = logging.getLogger("api_logger")


class EvaluationService:
    def __init__(self) -> None:
        self._ragas_service = RAGASService()

    async def record_query_evaluation(
        self,
        query: str,
        answer: str,
        docs_with_scores: List[Tuple[Document, float]],
        verification: VerificationResult | None,
        latency_ms: float,
    ) -> str | None:
        try:
            contexts = [doc.page_content for doc, _ in docs_with_scores]
            ragas_scores = self._ragas_service.score(query, answer, docs_with_scores)
            evaluation = EvaluationResult(
                query=query,
                answer=answer,
                contexts=contexts,
                faithfulness=verification.grounding_score if verification else None,
                precision=self._retrieval_precision(docs_with_scores),
                recall=ragas_scores.context_recall,
                context_recall=verification.source_coverage if verification else None,
                context_precision=ragas_scores.context_precision or self._context_precision(answer, contexts),
                answer_correctness=ragas_scores.overall,
                hallucination_rate=verification.hallucination_risk if verification else None,
                latency_ms=latency_ms,
            )
            db = MongoDBClient.get_db()
            await db["evaluations"].insert_one(evaluation.to_mongo_dict())
            return evaluation.evaluation_id
        except Exception as exc:
            logger.warning("Evaluation persistence failed: %s", exc)
            return None

    async def list_recent(self, limit: int = 50) -> List[Dict[str, Any]]:
        try:
            db = MongoDBClient.get_db()
            cursor = db["evaluations"].find({}).sort("created_at", -1).limit(limit)
            rows = await cursor.to_list(length=limit)
            output: List[Dict[str, Any]] = []
            for row in rows:
                row = dict(row)
                row["evaluation_id"] = str(row.pop("_id", ""))
                output.append(row)
            return output
        except Exception as exc:
            logger.warning("Evaluation listing failed: %s", exc)
            return []

    @staticmethod
    def _retrieval_precision(docs_with_scores: List[Tuple[Document, float]]) -> float:
        if not docs_with_scores:
            return 0.0
        useful = sum(1 for _, score in docs_with_scores if float(score) <= 0.0 or float(score) >= 0.5)
        return round(useful / len(docs_with_scores), 4)

    @staticmethod
    def _context_precision(answer: str, contexts: List[str]) -> float:
        answer_terms = {term for term in answer.lower().split() if len(term) > 2}
        if not answer_terms or not contexts:
            return 0.0
        context_terms = set()
        for context in contexts:
            context_terms.update(term for term in context.lower().split() if len(term) > 2)
        return round(len(answer_terms.intersection(context_terms)) / len(answer_terms), 4)


class LatencyTimer:
    def __init__(self) -> None:
        self._started = time.perf_counter()

    @property
    def elapsed_ms(self) -> float:
        return round((time.perf_counter() - self._started) * 1000, 3)
