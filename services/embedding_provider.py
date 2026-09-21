from abc import ABC, abstractmethod
from typing import List, Optional
import asyncio
import logging
from concurrent.futures import ThreadPoolExecutor

from tenacity import retry, wait_exponential, stop_after_attempt, retry_if_exception_type

from config.settings import get_settings

logger = logging.getLogger("api_logger")

_bge_executor: Optional[ThreadPoolExecutor] = None
_bge_model = None
_bge_model_lock = asyncio.Lock()


class EmbeddingProvider(ABC):
    @property
    @abstractmethod
    def version(self) -> str:
        pass

    @property
    @abstractmethod
    def dimension(self) -> int:
        pass

    @abstractmethod
    async def embed_documents(self, texts: List[str]) -> List[List[float]]:
        pass

    @abstractmethod
    async def embed_query(self, text: str) -> List[float]:
        pass


class OpenAIEmbeddingProvider(EmbeddingProvider):
    def __init__(self) -> None:
        from langchain_openai import OpenAIEmbeddings

        settings = get_settings()
        if not settings.openai_api_key:
            logger.warning("OpenAI API key is empty; embedding calls will fail until configured.")
        kwargs = {
            "model": settings.EMBEDDING_MODEL,
            "openai_api_key": settings.openai_api_key,
        }
        if settings.EMBEDDING_MODEL.startswith("text-embedding-3"):
            kwargs["dimensions"] = settings.EMBEDDING_DIMENSION
        self._client = OpenAIEmbeddings(**kwargs)
        self._settings = settings

    @property
    def version(self) -> str:
        return f"openai_{self._settings.EMBEDDING_MODEL}_d{self._settings.EMBEDDING_DIMENSION}"

    @property
    def dimension(self) -> int:
        return self._settings.EMBEDDING_DIMENSION

    @retry(
        wait=wait_exponential(multiplier=1, min=2, max=30),
        stop=stop_after_attempt(5),
        retry=retry_if_exception_type(Exception),
        reraise=True,
    )
    async def embed_documents(self, texts: List[str]) -> List[List[float]]:
        if not texts:
            return []
        return await self._client.aembed_documents(texts)

    @retry(
        wait=wait_exponential(multiplier=1, min=2, max=30),
        stop=stop_after_attempt(5),
        retry=retry_if_exception_type(Exception),
        reraise=True,
    )
    async def embed_query(self, text: str) -> List[float]:
        return await self._client.aembed_query(text)


class BGEM3EmbeddingProvider(EmbeddingProvider):
    def __init__(self) -> None:
        settings = get_settings()
        self._settings = settings
        self._model_name = settings.BGE_M3_MODEL
        self._inference_batch_size = settings.BGE_M3_INFERENCE_BATCH_SIZE
        self._max_length = settings.BGE_M3_MAX_LENGTH
        self._device = self._resolve_device(settings.EMBEDDING_DEVICE)
        self._use_fp16 = self._device == "cuda"
        logger.info(
            "BGE-M3 provider configured model=%s device=%s batch=%s",
            self._model_name,
            self._device,
            self._inference_batch_size,
        )

    @staticmethod
    def _resolve_device(configured: str) -> str:
        import torch

        normalized = (configured or "auto").lower()
        if normalized == "cpu":
            return "cpu"
        if normalized == "cuda":
            if not torch.cuda.is_available():
                raise RuntimeError("EMBEDDING_DEVICE=cuda but CUDA is not available.")
            return "cuda"
        return "cuda" if torch.cuda.is_available() else "cpu"

    async def _get_model(self):
        global _bge_model, _bge_executor
        async with _bge_model_lock:
            if _bge_model is not None:
                return _bge_model
            loop = asyncio.get_running_loop()

            def _load():
                global _bge_executor
                try:
                    from FlagEmbedding import BGEM3FlagModel
                except ImportError as exc:
                    raise RuntimeError(
                        "FlagEmbedding is required for EMBEDDING_PROVIDER=bge. "
                        "Install with: pip install FlagEmbedding"
                    ) from exc
                if _bge_executor is None:
                    _bge_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="bge-m3")
                model = BGEM3FlagModel(self._model_name, use_fp16=self._use_fp16)
                logger.info("BGE-M3 model loaded on %s", self._device)
                return model

            _bge_model = await loop.run_in_executor(None, _load)
            return _bge_model

    @property
    def version(self) -> str:
        return f"bge_m3_{self._model_name.replace('/', '_')}_d{self.dimension}"

    @property
    def dimension(self) -> int:
        return self._settings.BGE_M3_DIMENSION

    async def embed_documents(self, texts: List[str]) -> List[List[float]]:
        if not texts:
            return []
        model = await self._get_model()
        loop = asyncio.get_running_loop()
        executor = _bge_executor

        def _encode(batch: List[str]) -> List[List[float]]:
            output = model.encode(
                batch,
                batch_size=min(self._inference_batch_size, len(batch)),
                max_length=self._max_length,
            )
            dense = output.get("dense_vecs")
            if dense is None:
                raise RuntimeError("BGE-M3 encode did not return dense_vecs.")
            return dense.tolist()

        return await loop.run_in_executor(executor, _encode, texts)

    async def embed_query(self, text: str) -> List[float]:
        vectors = await self.embed_documents([text])
        if not vectors:
            raise RuntimeError("BGE-M3 query embedding returned no vectors.")
        return vectors[0]


def build_embedding_provider(provider_name: Optional[str] = None) -> EmbeddingProvider:
    settings = get_settings()
    name = (provider_name or settings.EMBEDDING_PROVIDER).lower().strip()
    if name in ("openai", "openai_embeddings"):
        return OpenAIEmbeddingProvider()
    if name in ("bge", "bge-m3", "bge_m3"):
        return BGEM3EmbeddingProvider()
    raise ValueError(f"Unsupported EMBEDDING_PROVIDER: {name}")
