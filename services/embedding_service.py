from langchain_openai import OpenAIEmbeddings
from config.settings import get_settings
from tenacity import retry, wait_exponential, stop_after_attempt, retry_if_exception_type
from typing import List
class EmbeddingService:
    def __init__(self) -> None:
        settings = get_settings()
        self.embeddings_model = OpenAIEmbeddings(openai_api_key=settings.openai_api_key, model='text-embedding-3-small')
    @retry(wait=wait_exponential(multiplier=1, min=2, max=10), stop=stop_after_attempt(3), retry=retry_if_exception_type(Exception), reraise=True)
    async def get_embeddings(self, texts: List[str]) -> List[List[float]]:
        return await self.embeddings_model.aembed_documents(texts)
    @retry(wait=wait_exponential(multiplier=1, min=2, max=10), stop=stop_after_attempt(3), retry=retry_if_exception_type(Exception), reraise=True)
    async def get_query_embedding(self, query: str) -> List[float]:
        return await self.embeddings_model.aembed_query(query)