import logging
from typing import List
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from config.settings import get_settings
from langchain_core.documents import Document

logger = logging.getLogger(__name__)

class ContextualEnrichmentService:
    def __init__(self):
        self.settings = get_settings()
        self.llm = None
        if self.settings.openai_api_key:
            try:
                self.llm = ChatOpenAI(
                    model="gpt-4o-mini", 
                    api_key=self.settings.openai_api_key,
                    temperature=0
                )
            except Exception as exc:
                logger.warning("Contextual enrichment LLM disabled: %s", exc)
        self.prompt = PromptTemplate.from_template(
            "You are an expert document analyzer. "
            "Here is the full document context:\n{parent_context}\n\n"
            "Here is a specific chunk from this document:\n{chunk_text}\n\n"
            "Write a very brief (1-2 sentences) context statement that situates this chunk within the overall document. "
            "Do not include the chunk itself, only the context."
        )
        
    async def enrich_chunk(self, chunk: Document, parent_context: str) -> Document:
        if self.llm is None:
            context = self._fallback_context(chunk.page_content, parent_context)
            enriched_content = f"Context: {context}\n\nContent: {chunk.page_content}"
            return Document(page_content=enriched_content, metadata=chunk.metadata)
        try:
            chain = self.prompt | self.llm
            response = await chain.ainvoke({
                "parent_context": parent_context[:5000],
                "chunk_text": chunk.page_content
            })
            context = response.content.strip()
            enriched_content = f"Context: {context}\n\nContent: {chunk.page_content}"
            enriched_doc = Document(page_content=enriched_content, metadata=chunk.metadata)
            return enriched_doc
        except Exception as e:
            logger.error(f"Failed to enrich chunk: {e}")
            return chunk
            
    async def enrich_chunks(self, chunks: List[Document], parent_context: str) -> List[Document]:
        enriched = []
        for c in chunks:
            enriched.append(await self.enrich_chunk(c, parent_context))
        return enriched

    @staticmethod
    def _fallback_context(chunk_text: str, parent_context: str) -> str:
        parent_snippet = " ".join(parent_context.split())[:240]
        chunk_snippet = " ".join(chunk_text.split())[:120]
        return f"Chunk '{chunk_snippet}' appears within parent context '{parent_snippet}'."
