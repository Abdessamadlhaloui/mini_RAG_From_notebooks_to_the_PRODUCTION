import logging
import uuid
from typing import List
from langchain_core.documents import Document
from services.embedding_service import EmbeddingService
from services.vector_db_service import VectorDBService
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
import asyncio
from config.settings import get_settings

logger = logging.getLogger(__name__)

class MultiVectorService:
    def __init__(self):
        self.settings = get_settings()
        self.embedding_service = EmbeddingService()
        self.vector_db = VectorDBService()
        self.llm = None
        if self.settings.openai_api_key:
            try:
                self.llm = ChatOpenAI(temperature=0, model="gpt-4o-mini", openai_api_key=self.settings.openai_api_key)
            except Exception as exc:
                logger.warning("MultiVector LLM disabled: %s", exc)
        
        self.summary_prompt = PromptTemplate.from_template(
            "Summarize the following text in one sentence for retrieval purposes:\n\n{text}"
        )
        self.qa_prompt = PromptTemplate.from_template(
            "Generate 3 hypothetical questions that this text could answer:\n\n{text}"
        )
        
    async def index_chunk_multi_vector(self, chunk: Document) -> None:
        tasks = [
            self._generate_summary(chunk.page_content),
            self._generate_questions(chunk.page_content)
        ]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        summary = results[0] if not isinstance(results[0], Exception) else ""
        questions = results[1] if not isinstance(results[1], Exception) else []
        
        docs_to_embed = [chunk]
        
        doc_id = chunk.metadata.get("parent_id") or chunk.metadata.get("chunk_id", str(uuid.uuid4()))
        
        if summary:
            meta = chunk.metadata.copy()
            meta["vector_type"] = "summary"
            meta["parent_chunk_id"] = doc_id
            docs_to_embed.append(Document(page_content=summary, metadata=meta))
            
        for q in questions:
            meta = chunk.metadata.copy()
            meta["vector_type"] = "question"
            meta["parent_chunk_id"] = doc_id
            docs_to_embed.append(Document(page_content=q, metadata=meta))
            
        embeddings = await self.embedding_service.embed_documents([d.page_content for d in docs_to_embed])
        version = self.embedding_service.version
        for doc in docs_to_embed:
            doc.metadata["embedding_version"] = version
        await self.vector_db.store_chunks(docs_to_embed, embeddings)
        
    async def _generate_summary(self, text: str) -> str:
        if self.llm is None:
            return self._fallback_summary(text)
        try:
            chain = self.summary_prompt | self.llm
            res = await chain.ainvoke({"text": text[:2000]})
            return res.content.strip()
        except Exception:
            return self._fallback_summary(text)
        
    async def _generate_questions(self, text: str) -> List[str]:
        if self.llm is None:
            return self._fallback_questions(text)
        try:
            chain = self.qa_prompt | self.llm
            res = await chain.ainvoke({"text": text[:2000]})
            questions = [q.strip() for q in res.content.split('\n') if q.strip() and '?' in q]
            return questions
        except Exception:
            return self._fallback_questions(text)

    @staticmethod
    def _fallback_summary(text: str) -> str:
        cleaned = " ".join(text.split())
        if not cleaned:
            return ""
        return cleaned[:240] + ("..." if len(cleaned) > 240 else "")

    @staticmethod
    def _fallback_questions(text: str) -> List[str]:
        words = [word.strip(".,;:!?") for word in text.split() if len(word.strip(".,;:!?")) > 4]
        unique = []
        for word in words:
            lowered = word.lower()
            if lowered not in unique:
                unique.append(lowered)
        unique = unique[:3]
        return [f"What is the role of {word} in this document?" for word in unique]
