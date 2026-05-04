"""
Application configuration management using pydantic-settings.
Loads values from .env and provides typed access throughout the application.
"""
from pydantic_settings import BaseSettings, SettingsConfigDict
from functools import lru_cache
from typing import List


class Settings(BaseSettings):
    """Central configuration object for the entire application."""

    # --- Application ---
    environment: str = "dev"

    # --- Authentication ---
    api_key: str = "super-secret-key"

    # --- OpenAI ---
    openai_api_key: str = ""

    # --- ChromaDB ---
    chroma_persist_directory: str = "./chroma_db"

    # --- RAG Pipeline ---
    chunk_size: int = 1000
    chunk_overlap: int = 200
    top_k: int = 4

    # --- MongoDB ---
    mongo_uri: str = "mongodb://localhost:27017/"
    mongo_db_name: str = "mini-rag"

    # --- CORS ---
    allowed_origins: str = "http://localhost:3000,http://localhost:8080"

    @property
    def cors_origins(self) -> List[str]:
        """Parses the comma-separated ALLOWED_ORIGINS string into a list."""
        return [origin.strip() for origin in self.allowed_origins.split(",") if origin.strip()]

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )


@lru_cache()
def get_settings() -> Settings:
    """Returns a cached singleton instance of the application settings."""
    return Settings()
