import asyncio
import os
import uuid
from typing import List

import aiofiles
from fastapi import HTTPException, UploadFile
from langchain_core.documents import Document
from werkzeug.utils import secure_filename

from services.document_processing_service import DocumentProcessingService

CACHE_DIR = "./cache/uploads"


def _sanitize_filename(original: str | None) -> str:
    if not original:
        return "unnamed_file"
    sanitized = secure_filename(original)
    return sanitized or "unnamed_file"


async def save_upload_file(upload_file: UploadFile) -> str:
    os.makedirs(CACHE_DIR, exist_ok=True)
    filename = f"{uuid.uuid4().hex}_{_sanitize_filename(upload_file.filename)}"
    file_path = os.path.join(CACHE_DIR, filename)
    try:
        async with aiofiles.open(file_path, "wb") as f:
            while chunk := await upload_file.read(1024 * 1024):
                await f.write(chunk)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Could not save file: {exc}") from exc
    return file_path


async def save_upload_file_tmp(upload_file: UploadFile) -> str:
    return await save_upload_file(upload_file)


async def load_document(file_path: str, filename: str | None = None, document_id: str | None = None) -> List[Document]:
    try:
        service = DocumentProcessingService()
        result = await service.process_file(file_path, filename, document_id=document_id)
        documents = result.documents
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Could not load document: {exc}") from exc

    normalized: List[Document] = []
    for index, document in enumerate(documents):
        metadata = dict(document.metadata or {})
        metadata["source"] = metadata.get("source") or metadata.get("source_filename") or filename or os.path.basename(file_path)
        metadata["document_part"] = index
        normalized.append(Document(page_content=document.page_content, metadata=metadata))
    return normalized


async def cleanup_temp_file(file_path: str) -> None:
    if not file_path:
        return
    try:
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(None, _remove_file_if_exists, file_path)
    except Exception:
        return


def _remove_file_if_exists(file_path: str) -> None:
    if os.path.isfile(file_path):
        os.remove(file_path)
