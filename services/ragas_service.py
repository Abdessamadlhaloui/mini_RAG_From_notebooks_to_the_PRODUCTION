import math
import re
from dataclasses import dataclass
from typing import Iterable, List, Tuple

from langchain_core.documents import Document


@dataclass(frozen=True)
class RAGASMetrics:
    faithfulness: float
    answer_relevancy: float
    context_precision: float
    context_recall: float
    answer_similarity: float
    overall: float


class RAGASService:
    def score(
        self,
        query: str,
        answer: str,
        docs_with_scores: List[Tuple[Document, float]],
    ) -> RAGASMetrics:
        if not docs_with_scores:
            return RAGASMetrics(0.0, 0.0, 0.0, 0.0, 0.0, 0.0)

        context_text = " ".join(doc.page_content for doc, _ in docs_with_scores)
        answer_tokens = set(self._tokens(answer))
        query_tokens = set(self._tokens(query))
        context_tokens = set(self._tokens(context_text))
        if not answer_tokens:
            return RAGASMetrics(0.0, 0.0, 0.0, 0.0, 0.0, 0.0)

        answer_relevancy = self._overlap_ratio(answer_tokens, query_tokens)
        faithfulness = self._overlap_ratio(answer_tokens, context_tokens)
        context_precision = self._source_precision(answer_tokens, docs_with_scores)
        context_recall = self._source_recall(answer_tokens, docs_with_scores)
        answer_similarity = self._jaccard(answer_tokens, query_tokens.union(context_tokens))
        overall = round(
            max(
                0.0,
                min(
                    1.0,
                    (faithfulness * 0.35)
                    + (answer_relevancy * 0.2)
                    + (context_precision * 0.2)
                    + (context_recall * 0.15)
                    + (answer_similarity * 0.1),
                ),
            ),
            4,
        )
        return RAGASMetrics(
            faithfulness=round(faithfulness, 4),
            answer_relevancy=round(answer_relevancy, 4),
            context_precision=round(context_precision, 4),
            context_recall=round(context_recall, 4),
            answer_similarity=round(answer_similarity, 4),
            overall=overall,
        )

    @staticmethod
    def _tokens(text: str) -> List[str]:
        return [token for token in re.findall(r"[a-zA-Z0-9_]+", text.lower()) if len(token) > 2]

    @staticmethod
    def _overlap_ratio(lhs: set[str], rhs: set[str]) -> float:
        if not lhs or not rhs:
            return 0.0
        return len(lhs.intersection(rhs)) / max(len(lhs), 1)

    def _source_precision(self, answer_tokens: set[str], docs_with_scores: List[Tuple[Document, float]]) -> float:
        if not answer_tokens:
            return 0.0
        matched = 0
        for doc, _ in docs_with_scores:
            doc_tokens = set(self._tokens(doc.page_content))
            if len(answer_tokens.intersection(doc_tokens)) / max(len(answer_tokens), 1) >= 0.15:
                matched += 1
        return matched / max(len(docs_with_scores), 1)

    def _source_recall(self, answer_tokens: set[str], docs_with_scores: List[Tuple[Document, float]]) -> float:
        if not docs_with_scores:
            return 0.0
        covered_tokens = set()
        for doc, _ in docs_with_scores:
            covered_tokens.update(self._tokens(doc.page_content))
        return self._overlap_ratio(answer_tokens, covered_tokens)

    @staticmethod
    def _jaccard(lhs: set[str], rhs: set[str]) -> float:
        if not lhs or not rhs:
            return 0.0
        return len(lhs.intersection(rhs)) / max(len(lhs.union(rhs)), 1)
