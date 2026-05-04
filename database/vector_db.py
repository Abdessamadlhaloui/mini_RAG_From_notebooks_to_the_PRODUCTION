"""
ChromaDB singleton client.
Abstracts the vector database connection so the backend can be swapped
(e.g., to Qdrant, Pinecone, or MongoDB Atlas Vector Search) by changing
only this module.
"""
import chromadb
from chromadb.config import Settings as ChromaSettings
from config.settings import get_settings
import logging

logger = logging.getLogger("api_logger")


class ChromaDBClient:
    """Thread-safe singleton for the ChromaDB persistent client."""

    _instance = None

    @classmethod
    def get_client(cls) -> chromadb.ClientAPI:
        """Returns the singleton ChromaDB client, creating it on first call."""
        if cls._instance is None:
            settings = get_settings()
            cls._instance = chromadb.PersistentClient(
                path=settings.chroma_persist_directory,
                settings=ChromaSettings(anonymized_telemetry=False),
            )
            logger.info(
                "ChromaDB PersistentClient initialized at '%s'.",
                settings.chroma_persist_directory,
            )
        return cls._instance

    @classmethod
    def reset(cls) -> None:
        """Resets the singleton (useful for testing)."""
        cls._instance = None
