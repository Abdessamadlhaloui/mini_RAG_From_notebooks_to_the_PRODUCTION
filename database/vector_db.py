import logging
from typing import Any

from config.settings import get_settings

logger = logging.getLogger("api_logger")


class ChromaDBClient:
    _instance: Any = None

    @classmethod
    def get_client(cls) -> Any:
        if cls._instance is None:
            try:
                import chromadb
                from chromadb.config import Settings as ChromaSettings
            except Exception as exc:
                raise RuntimeError(f"ChromaDB is unavailable in this runtime: {exc}") from exc
            settings = get_settings()
            cls._instance = chromadb.PersistentClient(
                path=settings.chroma_persist_directory,
                settings=ChromaSettings(anonymized_telemetry=False),
            )
            logger.info("ChromaDB PersistentClient initialized at '%s'.", settings.chroma_persist_directory)
        return cls._instance

    @classmethod
    def reset(cls) -> None:
        cls._instance = None
