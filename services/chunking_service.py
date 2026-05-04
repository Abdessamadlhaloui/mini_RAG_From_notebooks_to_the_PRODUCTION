"""
ChunkingService — splits LangChain Documents into smaller chunks
while preserving metadata across each chunk.

Synchronous text splitting is dispatched to a thread-pool executor
to avoid blocking the async event loop.
"""
import asyncio
from functools import partial
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document
from config.settings import get_settings
from typing import List


class ChunkingService:
    """Splits documents into overlapping text chunks for embedding."""

    def __init__(self) -> None:
        settings = get_settings()
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=settings.chunk_size,
            chunk_overlap=settings.chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", " ", ""],
        )

    def _split_sync(self, documents: List[Document]) -> List[Document]:
        """CPU-bound splitting — runs inside an executor."""
        return self.text_splitter.split_documents(documents)

    async def chunk_documents(self, documents: List[Document]) -> List[Document]:
        """
        Splits documents into chunks asynchronously.
        Delegates to a thread-pool so the event loop is never blocked.
        """
        loop = asyncio.get_running_loop()
        chunks = await loop.run_in_executor(
            None, partial(self._split_sync, documents)
        )
        return chunks
