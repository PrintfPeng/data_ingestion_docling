from __future__ import annotations
import os
from typing import List
from dotenv import load_dotenv
from langchain_google_genai import GoogleGenerativeAIEmbeddings

class EmbeddingService:
    def __init__(self):
        self._model_name = "models/text-embedding-004"
        self.client = self._get_client()

    def _get_client(self) -> GoogleGenerativeAIEmbeddings:
        load_dotenv(override=True)
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise RuntimeError("GOOGLE_API_KEY is not set in .env")
        
        return GoogleGenerativeAIEmbeddings(
            model=self._model_name,
            google_api_key=api_key.strip(),
        )

    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        if not texts: return []
        return self.client.embed_documents(texts)

    def embed_query(self, text: str) -> List[float]:
        return self.client.embed_query(text)