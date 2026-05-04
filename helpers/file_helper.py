"""
File helper utilities for secure upload handling and document loading.

Fixes applied:
- Path traversal: uses werkzeug.utils.secure_filename + UUID prefix.
- Async blocking: wraps synchronous PyPDFLoader / TextLoader in run_in_executor.
- File cleanup is handled externally via BackgroundTasks (not in this module).
"""
import os
import uuid
import asyncio
import aiofiles
from functools import partial
from fastapi import UploadFile, HTTPException
from werkzeug.utils import secure_filename
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain.docstore.document import Document
from typing import List

CACHE_DIR = "./cache/uploads"


def _sanitize_filename(original: str) -> str:
    """
    Produces a safe, unique filename to prevent path-traversal attacks.
    Example: '../../../etc/passwd' -> 'a1b2c3d4_passwd'
    """
    safe_name = secure_filename(original)
    if not safe_name:
        safe_name = "unnamed_file"
    return f"{uuid.uuid4().hex[:8]}_{safe_name}"


async def save_upload_file_tmp(upload_file: UploadFile) -> str:
    """
    Saves an UploadFile asynchronously to the cache directory with a sanitized name.

    Returns:
        The absolute path to the saved file.
    """
    try:
        os.makedirs(CACHE_DIR, exist_ok=True)
        safe_name = _sanitize_filename(upload_file.filename or "unknown")
        file_path = os.path.join(CACHE_DIR, safe_name)

        async with aiofiles.open(file_path, "wb") as out_file:
            while content := await upload_file.read(1024 * 1024):
                await out_file.write(content)

        return file_path
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to save uploaded file: {exc}",
        )


def _sync_load_pdf(file_path: str) -> List[Document]:
    """Synchronous PDF loading — called inside run_in_executor."""
    loader = PyPDFLoader(file_path)
    return loader.load()


def _sync_load_txt(file_path: str) -> List[Document]:
    """Synchronous text file loading — called inside run_in_executor."""
    loader = TextLoader(file_path, encoding="utf-8")
    return loader.load()


async def load_document(file_path: str, original_filename: str) -> List[Document]:
    """
    Loads a document using the appropriate LangChain loader.

    Synchronous loaders are dispatched to a thread-pool executor to avoid
    blocking the async event loop.

    Note: Temporary file cleanup is NOT done here — it is delegated to a
    FastAPI BackgroundTask in the route layer for reliability.
    """
    loop = asyncio.get_running_loop()

    if original_filename.lower().endswith(".pdf"):
        docs = await loop.run_in_executor(None, partial(_sync_load_pdf, file_path))
    elif original_filename.lower().endswith(".txt"):
        docs = await loop.run_in_executor(None, partial(_sync_load_txt, file_path))
    else:
        raise ValueError(f"Unsupported file extension for '{original_filename}'")

    # Enrich metadata with the original filename
    for doc in docs:
        doc.metadata["source_filename"] = original_filename

    return docs


def cleanup_temp_file(file_path: str) -> None:
    """
    Removes a temporary file from disk.
    Intended to be called from a FastAPI BackgroundTask after the response is sent.
    """
    try:
        if os.path.exists(file_path):
            os.remove(file_path)
    except OSError:
        pass  # Best-effort cleanup; logged at the route level if needed
