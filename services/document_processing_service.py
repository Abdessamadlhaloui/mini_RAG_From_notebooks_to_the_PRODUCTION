import asyncio
import logging
import os
import tempfile
import uuid
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from langchain_core.documents import Document

from config.settings import get_settings
from models.document_model import DocumentModel
from models.document_structure_model import DocumentStructure, FormulaData, ImageData, SectionNode, TableData
from services.formula_service import FormulaService
from services.image_captioning_service import ImageCaptioningService
from services.layout_service import LayoutService
from services.ocr_service import OCRService
from services.table_extraction_service import TableExtractionService

logger = logging.getLogger("api_logger")


@dataclass(frozen=True)
class DocumentProcessingResult:
    document: DocumentModel
    structure: DocumentStructure
    documents: List[Document]


class DocumentProcessingService:
    def __init__(
        self,
        ocr_service: OCRService | None = None,
        layout_service: LayoutService | None = None,
        table_service: TableExtractionService | None = None,
        formula_service: FormulaService | None = None,
        image_captioning_service: ImageCaptioningService | None = None,
    ) -> None:
        self._settings = get_settings()
        self.ocr_service = ocr_service or OCRService()
        self.layout_service = layout_service or LayoutService()
        self.table_service = table_service or TableExtractionService()
        self.formula_service = formula_service or FormulaService()
        self.image_captioning_service = image_captioning_service or ImageCaptioningService()

    async def process_file(self, file_path: str, filename: str | None = None, document_id: str | None = None) -> DocumentProcessingResult:
        ext = os.path.splitext(file_path)[1].lower()
        source_filename = filename or os.path.basename(file_path)
        document_id = document_id or str(uuid.uuid4())

        if ext == ".pdf" and self._settings.document_enable_pdf_processing:
            return await self._process_pdf(file_path, source_filename, document_id)
        return await self._process_text(file_path, source_filename, document_id)

    async def _process_text(self, file_path: str, source_filename: str, document_id: str) -> DocumentProcessingResult:
        from langchain_community.document_loaders import TextLoader

        loader = TextLoader(file_path, encoding="utf-8", autodetect_encoding=True)
        loop = asyncio.get_running_loop()
        docs = await loop.run_in_executor(None, loader.load)
        normalized: List[Document] = []
        for index, doc in enumerate(docs):
            metadata = self._base_metadata(document_id, source_filename)
            metadata.update(doc.metadata or {})
            metadata.update(
                {
                    "page": metadata.get("page", index + 1),
                    "section_id": f"text_{index}",
                    "hierarchy_level": "document",
                    "chunk_type": "text",
                }
            )
            normalized.append(Document(page_content=doc.page_content, metadata=metadata))
        structure = DocumentStructure(
            document_id=document_id,
            title=source_filename,
            author=None,
            language="en",
            page_count=1,
            metadata={"source_filename": source_filename, "file_type": "txt"},
            sections=[
                SectionNode(
                    section_id="text_0",
                    level="paragraph",
                    title=source_filename,
                    content=normalized[0].page_content if normalized else "",
                    page=1,
                )
            ],
        )
        model = self._build_document_model(document_id, source_filename, normalized, structure, has_ocr=False)
        return DocumentProcessingResult(document=model, structure=structure, documents=normalized)

    async def _process_pdf(self, file_path: str, source_filename: str, document_id: str) -> DocumentProcessingResult:
        try:
            import fitz
        except ImportError as exc:
            logger.warning("PyMuPDF is unavailable; falling back to plain PDF text extraction: %s", exc)
            return await self._process_pdf_fallback(file_path, source_filename, document_id)

        loop = asyncio.get_running_loop()
        doc = await loop.run_in_executor(None, fitz.open, file_path)
        try:
            structure = self.layout_service.extract_structure(file_path, document_id) if self._settings.document_enable_layout_parsing else DocumentStructure(document_id=document_id)
            table_data = self.table_service.extract_tables(file_path) if self._settings.document_enable_table_extraction else []
            document_metadata = self._extract_pdf_metadata(doc, source_filename)
            documents: List[Document] = []
            page_docs = await self._extract_page_documents(doc, document_id, source_filename)
            documents.extend(page_docs)
            artifacts = await self._extract_pdf_artifacts(doc, document_id, source_filename, structure)
            documents.extend(artifacts)

            if table_data:
                structure.tables = [
                    TableData(
                        table_id=table.get("table_id") or f"table_{index}",
                        page=int(table.get("page") or 1),
                        title=table.get("title"),
                        rows=table.get("rows") or [],
                        raw_json=table.get("raw_json") or "[]",
                    )
                    for index, table in enumerate(table_data)
                ]
                for table in structure.tables:
                    documents.append(
                        Document(
                            page_content=table.raw_json,
                            metadata=self._base_metadata(document_id, source_filename)
                            | {
                                "page": table.page,
                                "table_id": table.table_id,
                                "chunk_type": "table",
                                "hierarchy_level": "artifact",
                                "section_id": f"table_{table.table_id}",
                            },
                        )
                    )

            structure.metadata.update(document_metadata)
            model = self._build_document_model(
                document_id=document_id,
                source_filename=source_filename,
                documents=documents,
                structure=structure,
                has_ocr=any(doc.metadata.get("has_ocr") for doc in documents),
                table_count=len(structure.tables),
                image_count=len(structure.images),
                formula_count=len(structure.formulas),
            )
            return DocumentProcessingResult(document=model, structure=structure, documents=documents)
        finally:
            doc.close()

    async def _process_pdf_fallback(self, file_path: str, source_filename: str, document_id: str) -> DocumentProcessingResult:
        from langchain_community.document_loaders import PyPDFLoader

        loader = PyPDFLoader(file_path)
        loop = asyncio.get_running_loop()
        docs = await loop.run_in_executor(None, loader.load)
        normalized: List[Document] = []
        for index, doc in enumerate(docs):
            metadata = self._base_metadata(document_id, source_filename)
            metadata.update(doc.metadata or {})
            metadata.update({"page": metadata.get("page", index + 1), "section_id": f"page_{index + 1}", "chunk_type": "text"})
            normalized.append(Document(page_content=doc.page_content, metadata=metadata))
        structure = DocumentStructure(
            document_id=document_id,
            title=source_filename,
            page_count=len(normalized),
            metadata={"source_filename": source_filename, "file_type": "pdf"},
        )
        model = self._build_document_model(document_id, source_filename, normalized, structure, has_ocr=False)
        return DocumentProcessingResult(document=model, structure=structure, documents=normalized)

    async def _extract_page_documents(self, pdf_doc: Any, document_id: str, source_filename: str) -> List[Document]:
        try:
            import fitz
        except ImportError:
            return []

        documents: List[Document] = []
        for page_number in range(len(pdf_doc)):
            page = pdf_doc[page_number]
            text = page.get_text("text").strip()
            page_meta = self._base_metadata(document_id, source_filename)
            page_meta.update(
                {
                    "page": page_number + 1,
                    "section_id": f"page_{page_number + 1}",
                    "page_label": page_number + 1,
                    "hierarchy_level": "page",
                    "chunk_type": "text",
                }
            )
            if text:
                documents.append(Document(page_content=text, metadata=page_meta))
            elif self._settings.document_enable_ocr:
                with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
                    pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))
                    pix.save(tmp.name)
                    ocr_result = self.ocr_service.extract_text(tmp.name)
                os.unlink(tmp.name)
                ocr_text = (ocr_result.get("text") or "").strip()
                if ocr_text:
                    page_meta["has_ocr"] = True
                    page_meta["ocr_confidence"] = ocr_result.get("confidence")
                    documents.append(Document(page_content=ocr_text, metadata=page_meta))
        return documents

    async def _extract_pdf_artifacts(
        self,
        pdf_doc: Any,
        document_id: str,
        source_filename: str,
        structure: DocumentStructure,
    ) -> List[Document]:
        if not self._settings.document_enable_image_extraction:
            return []
        try:
            import fitz
        except ImportError:
            return []

        documents: List[Document] = []
        extracted_dir = os.path.join(tempfile.gettempdir(), "mini_rag_extracted")
        os.makedirs(extracted_dir, exist_ok=True)
        for page_number in range(len(pdf_doc)):
            page = pdf_doc[page_number]
            image_list = page.get_images(full=True)
            for image_index, image in enumerate(image_list):
                xref = image[0]
                extracted = pdf_doc.extract_image(xref)
                image_bytes = extracted.get("image")
                if not image_bytes:
                    continue
                ext = extracted.get("ext", "png")
                image_path = os.path.join(extracted_dir, f"{document_id}_p{page_number + 1}_{image_index}.{ext}")
                with open(image_path, "wb") as handle:
                    handle.write(image_bytes)
                caption = ""
                if self._settings.document_enable_image_captioning:
                    caption = self.image_captioning_service.generate_caption(image_path) or ""
                ocr_text = ""
                if self._settings.document_enable_ocr:
                    ocr_result = self.ocr_service.extract_text(image_path)
                    ocr_text = (ocr_result.get("text") or "").strip()
                formula_text = ""
                if self._settings.document_enable_formula_recognition:
                    formula_text = self.formula_service.extract_formula(image_path) or ""
                content_parts = [part for part in [caption, ocr_text, formula_text] if part]
                if not content_parts:
                    continue
                rects = page.get_image_rects(xref)
                bbox = None
                if rects:
                    first_rect = rects[0]
                    bbox = [float(first_rect.x0), float(first_rect.y0), float(first_rect.x1), float(first_rect.y1)]
                structure.images.append(
                    ImageData(
                        image_id=f"img_{page_number + 1}_{image_index}",
                        page=page_number + 1,
                        position=bbox,
                        image_path=image_path,
                        caption=caption or None,
                    )
                )
                if formula_text:
                    structure.formulas.append(
                        FormulaData(
                            formula_id=f"formula_{page_number + 1}_{image_index}",
                            page=page_number + 1,
                            latex=formula_text,
                            rendered_text=caption or ocr_text or None,
                        )
                    )
                documents.append(
                    Document(
                        page_content="\n".join(content_parts),
                        metadata=self._base_metadata(document_id, source_filename)
                        | {
                            "page": page_number + 1,
                            "image_index": image_index,
                            "image_path": image_path,
                            "caption": caption,
                            "ocr_text": ocr_text,
                            "formula_text": formula_text,
                            "chunk_type": "image",
                            "hierarchy_level": "artifact",
                            "section_id": f"image_{page_number + 1}_{image_index}",
                        },
                    )
                )
        return documents

    def _extract_pdf_metadata(self, pdf_doc: Any, source_filename: str) -> Dict[str, Any]:
        raw = pdf_doc.metadata or {}
        return {
            "source_filename": source_filename,
            "file_type": "pdf",
            "title": raw.get("title") or source_filename,
            "author": raw.get("author"),
            "language": raw.get("language"),
            "creation_date": raw.get("creationDate"),
            "page_count": len(pdf_doc),
        }

    def _build_document_model(
        self,
        document_id: str,
        source_filename: str,
        documents: List[Document],
        structure: DocumentStructure,
        has_ocr: bool,
        table_count: int = 0,
        image_count: int = 0,
        formula_count: int = 0,
    ) -> DocumentModel:
        metadata = dict(structure.metadata or {})
        return DocumentModel(
            document_id=document_id,
            filename=source_filename,
            file_type=metadata.get("file_type"),
            chunks_processed=len(documents),
            status="success",
            page_count=structure.page_count or len(documents),
            author=structure.author or metadata.get("author"),
            language=structure.language or metadata.get("language"),
            creation_date=structure.creation_date or metadata.get("creation_date"),
            document_title=structure.title or metadata.get("title"),
            has_ocr=has_ocr,
            ocr_confidence=self._derive_ocr_confidence(documents),
            tables_count=table_count,
            images_count=image_count,
            formulas_count=formula_count,
            structure_id=document_id,
            embedding_version=None,
        )

    @staticmethod
    def _derive_ocr_confidence(documents: List[Document]) -> Optional[float]:
        confidences = [float(doc.metadata.get("ocr_confidence")) for doc in documents if doc.metadata.get("ocr_confidence") is not None]
        if not confidences:
            return None
        return round(sum(confidences) / len(confidences), 4)

    @staticmethod
    def _base_metadata(document_id: str, source_filename: str) -> Dict[str, Any]:
        return {
            "document_id": document_id,
            "source_filename": source_filename,
            "filename": source_filename,
        }
