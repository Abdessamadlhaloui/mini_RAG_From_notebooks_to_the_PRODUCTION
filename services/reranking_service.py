import asyncio
import logging
from concurrent.futures import ThreadPoolExecutor
from typing import List, Tuple

from langchain_core.documents import Document

from config.settings import get_settings

logger = logging.getLogger("api_logger")

_rerank_executor: ThreadPoolExecutor | None = None
_cross_encoder = None


class RerankingService:
    def __init__(self) -> None:
        self._settings = get_settings()

    async def rerank(
        self,
        query: str,
        candidates: List[Tuple[Document, float]],
        top_k: int,
    ) -> List[Tuple[Document, float]]:
        if not candidates:
            return []
        if not self._settings.RETRIEVAL_ENABLE_RERANK or len(candidates) <= top_k:
            return candidates[:top_k]
        model = await self._get_model()
        loop = asyncio.get_running_loop()
        executor = _rerank_executor
        pairs = [[query, doc.page_content] for doc, _ in candidates]

        def _predict():
            scores = model.predict(pairs, batch_size=self._settings.CROSS_ENCODER_BATCH_SIZE, show_progress_bar=False)
            return [float(score) for score in scores]

        scores = await loop.run_in_executor(executor, _predict)
        ranked = sorted(zip(candidates, scores), key=lambda item: item[1], reverse=True)
        output: List[Tuple[Document, float]] = []
        for (doc, _), score in ranked[:top_k]:
            metadata = dict(doc.metadata)
            metadata["rerank_score"] = score
            output.append((Document(page_content=doc.page_content, metadata=metadata), -score))
        output.sort(key=lambda item: item[1])
        return output

    async def _get_model(self):
        global _cross_encoder, _rerank_executor
        if _cross_encoder is not None:
            return _cross_encoder
        loop = asyncio.get_running_loop()

        def _load():
            global _rerank_executor
            import torch
            from sentence_transformers import CrossEncoder

            settings = get_settings()
            device = settings.CROSS_ENCODER_DEVICE.lower()
            if device == "auto":
                resolved = "cuda" if torch.cuda.is_available() else "cpu"
            elif device == "cuda":
                if not torch.cuda.is_available():
                    raise RuntimeError("CROSS_ENCODER_DEVICE=cuda but CUDA is not available.")
                resolved = "cuda"
            else:
                resolved = "cpu"
            if _rerank_executor is None:
                _rerank_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="cross-encoder")
            model = CrossEncoder(settings.CROSS_ENCODER_MODEL, device=resolved)
            logger.info("Cross-encoder loaded model=%s device=%s", settings.CROSS_ENCODER_MODEL, resolved)
            return model

        _cross_encoder = await loop.run_in_executor(None, _load)
        return _cross_encoder
