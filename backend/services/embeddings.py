from __future__ import annotations

import os
from typing import List

from dotenv import load_dotenv
from langchain_google_genai import GoogleGenerativeAIEmbeddings

# ใช้โมเดล embeddings ของ Gemini
_EMBEDDING_MODEL_NAME = "models/text-embedding-004"

# เก็บ client แบบ singleton
_embeddings_client: GoogleGenerativeAIEmbeddings | None = None


def _load_api_key() -> str:
    """
    โหลด GOOGLE_API_KEY โดย 'บังคับ' ให้ .env ทับค่าของเดิม (ฆ่า Ghost Key)
    """
    # override=True = ให้ค่าจาก .env ทับของเดิมใน env
    load_dotenv(override=True)

    api_key = os.getenv("GOOGLE_API_KEY")

    if not api_key or not api_key.strip():
        raise RuntimeError(
            "GOOGLE_API_KEY is not set. "
            "Please add it to your .env file or environment."
        )

    api_key = api_key.strip()

    # ใส่กลับเข้า os.environ ให้ lib ตัวอื่นใช้ได้แน่นอน
    os.environ["GOOGLE_API_KEY"] = api_key

    return api_key


def get_embedding_client() -> GoogleGenerativeAIEmbeddings:
    """
    คืน client สำหรับสร้าง embedding ด้วย Gemini (ใช้ singleton ป้องกันสร้างซ้ำ)
    """
    global _embeddings_client

    if _embeddings_client is None:
        api_key = _load_api_key()

        # ส่ง key ตรง ๆ ให้ชัดเจน
        _embeddings_client = GoogleGenerativeAIEmbeddings(
            model=_EMBEDDING_MODEL_NAME,
            google_api_key=api_key,
        )

    return _embeddings_client


def embed_texts(texts: List[str]) -> List[List[float]]:
    """
    helper สำหรับ embed เป็น batch จาก list ของข้อความ
    """
    if not texts:
        return []

    client = get_embedding_client()
    return client.embed_documents(texts)


def embed_query(text: str) -> List[float]:
    """
    helper สำหรับ embed ข้อความเดี่ยว (เช่นใช้ตอน similarity search)
    """
    client = get_embedding_client()
    return client.embed_query(text)
