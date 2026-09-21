import logging
import math
import re
from collections import Counter
from typing import Iterable, List, Tuple

from langchain_core.documents import Document

from config.settings import get_settings
from schemas.rag_schema import Citation, VerificationResult

logger = logging.getLogger("api_logger")


class RAGQualityService:
    def __init__(self) -> None:
        self._settings = get_settings()

    def build_citations(self, docs_with_scores: List[Tuple[Document, float]]) -> List[Citation]:
        citations: List[Citation] = []
        for index, (doc, score) in enumerate(docs_with_scores):
            metadata = dict(doc.metadata or {})
            excerpt = self._trim_excerpt(metadata.get("child_excerpt") or doc.page_content)
            citations.append(
                Citation(
                    citation_id=f"S{index + 1}",
                    source_index=index,
                    chunk_id=self._string_or_none(metadata.get("chunk_id") or metadata.get("parent_chunk_id")),
                    document_id=self._string_or_none(metadata.get("document_id")),
                    source_filename=self._string_or_none(metadata.get("source_filename") or metadata.get("filename")),
                    excerpt=excerpt,
                    score=float(score) if score is not None else None,
                )
            )
        return citations

    def verify_answer(
        self,
        query: str,
        answer: str,
        docs_with_scores: List[Tuple[Document, float]],
    ) -> VerificationResult:
        if not answer.strip() or not docs_with_scores:
            return VerificationResult(grounded=False, grounding_score=0.0, hallucination_risk=1.0, source_coverage=0.0)
        context = "\n ".join(doc.page_content for doc, _ in docs_with_scores)
        answer_sentences = self._sentences(answer)
        if not answer_sentences:
            return VerificationResult(grounded=False, grounding_score=0.0, hallucination_risk=1.0, source_coverage=0.0)
        context_tokens = set(self._tokens(context))
        query_tokens = set(self._tokens(query))
        supported = 0
        unsupported_claims: List[str] = []
        for sentence in answer_sentences:
            sentence_tokens = [token for token in self._tokens(sentence) if token not in query_tokens]
            if not sentence_tokens:
                supported += 1
                continue
            overlap = len(set(sentence_tokens).intersection(context_tokens)) / max(len(set(sentence_tokens)), 1)
            if overlap >= self._settings.GENERATION_MIN_GROUNDING_SCORE:
                supported += 1
            else:
                unsupported_claims.append(sentence)
        grounding_score = supported / max(len(answer_sentences), 1)
        coverage = self._source_coverage(answer, docs_with_scores)
        hallucination_risk = max(0.0, min(1.0, 1.0 - ((grounding_score * 0.75) + (coverage * 0.25))))
        grounded = grounding_score >= 0.7 and hallucination_risk <= 0.4
        return VerificationResult(
            grounded=grounded,
            grounding_score=round(grounding_score, 4),
            hallucination_risk=round(hallucination_risk, 4),
            source_coverage=round(coverage, 4),
            unsupported_claims=unsupported_claims[:5],
        )

    def confidence_score(self, verification: VerificationResult, docs_with_scores: List[Tuple[Document, float]]) -> float:
        if not docs_with_scores:
            return 0.0
        retrieval_signal = self._retrieval_signal([score for _, score in docs_with_scores])
        confidence = (verification.grounding_score * 0.55) + (verification.source_coverage * 0.2) + (retrieval_signal * 0.25)
        return round(max(0.0, min(1.0, confidence)), 4)

    def append_inline_citations(self, answer: str, citations: List[Citation]) -> str:
        if not self._settings.GENERATION_ENABLE_CITATIONS or not citations or not answer.strip():
            return answer
        if re.search(r"\[S\d+\]", answer):
            return answer
        sentence_parts = re.split(r"(?<=[.!?])\s+", answer.strip())
        output: List[str] = []
        for index, sentence in enumerate(sentence_parts):
            if not sentence:
                continue
            citation = citations[min(index, len(citations) - 1)]
            output.append(f"{sentence} [{citation.citation_id}]")
        return " ".join(output)

    @staticmethod
    def _tokens(text: str) -> List[str]:
        return [token for token in re.findall(r"[a-zA-Z0-9_]+", text.lower()) if len(token) > 2]

    @staticmethod
    def _sentences(text: str) -> List[str]:
        return [sentence.strip() for sentence in re.split(r"(?<=[.!?])\s+", text) if sentence.strip()]

    @staticmethod
    def _trim_excerpt(text: str, limit: int = 500) -> str:
        normalized = re.sub(r"\s+", " ", text or "").strip()
        if len(normalized) <= limit:
            return normalized
        return normalized[: limit - 3].rstrip() + "..."

    @staticmethod
    def _string_or_none(value: object) -> str | None:
        if value is None:
            return None
        text = str(value)
        return text if text else None

    def _source_coverage(self, answer: str, docs_with_scores: List[Tuple[Document, float]]) -> float:
        answer_tokens = set(self._tokens(answer))
        if not answer_tokens:
            return 0.0
        covered_sources = 0
        for doc, _ in docs_with_scores:
            source_tokens = set(self._tokens(doc.page_content))
            if len(answer_tokens.intersection(source_tokens)) / max(len(answer_tokens), 1) >= 0.1:
                covered_sources += 1
        return covered_sources / max(len(docs_with_scores), 1)

    @staticmethod
    def _retrieval_signal(scores: Iterable[float]) -> float:
        values = [abs(float(score)) for score in scores]
        if not values:
            return 0.0
        average = sum(values) / len(values)
        return 1.0 / (1.0 + math.exp(min(average, 20.0) - 5.0))


class RetrievalExplainabilityService:
    def explain(self, docs_with_scores: List[Tuple[Document, float]]) -> List[dict]:
        explanations: List[dict] = []
        for rank, (doc, score) in enumerate(docs_with_scores, start=1):
            metadata = dict(doc.metadata or {})
            explanations.append(
                {
                    "rank": rank,
                    "score": float(score),
                    "retrieval_source": metadata.get("retrieval_source", "dense"),
                    "chunk_id": metadata.get("chunk_id"),
                    "document_id": metadata.get("document_id"),
                    "source_filename": metadata.get("source_filename") or metadata.get("filename"),
                    "rerank_score": metadata.get("rerank_score"),
                    "parent_context": bool(metadata.get("parent_context")),
                }
            )
        return explanations
