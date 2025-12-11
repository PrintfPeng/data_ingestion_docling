from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Dict, Tuple
import logging
import warnings

from fastapi import HTTPException
from langchain_community.vectorstores import Chroma
from langchain_google_genai._common import GoogleGenerativeAIError
# ใช้ Document type ของ langchain เพื่อความ compatible
from langchain_core.documents import Document

from .chunking import Chunk
from .embeddings import get_embedding_client

# -----------------------------------------------------------
# Setup
# -----------------------------------------------------------
logger = logging.getLogger(__name__)

# Fallback error import
try:
    from chromadb.errors import InternalError as ChromaInternalError
except Exception:
    ChromaInternalError = Exception

# -----------------------------------------------------------
# Configuration
# -----------------------------------------------------------

CHROMA_DIR = "chroma_db"
COLLECTION_NAME = "documents"

# Cache vectordb ตาม (persist_directory, collection_name)
_vectordb_cache: Dict[Tuple[str, str], Chroma] = {}


# -----------------------------------------------------------
# Helper: Chroma DB Management
# -----------------------------------------------------------

def _cache_key(persist_directory: str, collection_name: str) -> Tuple[str, str]:
    return (str(Path(persist_directory).resolve()), collection_name)


def get_vector_store(
    persist_directory: str = CHROMA_DIR,
    collection_name: str = COLLECTION_NAME,
    force_recreate: bool = False,
) -> Chroma:
    """
    คืน Chroma vector store ที่ผูกกับ Gemini embeddings
    - มี cache ใน process เดียวกันเพื่อไม่ต้องสร้างใหม่ทุกครั้ง
    """
    persist_path = Path(persist_directory)
    persist_path.mkdir(parents=True, exist_ok=True)
    key = _cache_key(persist_directory, collection_name)

    if not force_recreate and key in _vectordb_cache:
        return _vectordb_cache[key]

    embeddings = get_embedding_client()

    try:
        # Suppress deprecation warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            vectordb = Chroma(
                collection_name=collection_name,
                embedding_function=embeddings,
                persist_directory=str(persist_path),
            )
    except Exception as e:
        logger.exception("[vector_store] Failed to init Chroma: %s", e)
        raise HTTPException(
            status_code=500,
            detail="ไม่สามารถเชื่อมต่อ Vector DB ได้ โปรดตรวจสอบการติดตั้ง"
        ) from e

    _vectordb_cache[key] = vectordb
    return vectordb


def _normalize_metadata(md: dict) -> dict:
    """แปลงค่า complex types เป็น string เพื่อให้ Chroma เก็บได้"""
    simple: dict = {}
    for k, v in (md or {}).items():
        if isinstance(v, (str, int, float, bool)) or v is None:
            simple[k] = v
        else:
            try:
                simple[k] = str(v)
            except Exception:
                simple[k] = repr(v)
    return simple


def _raise_embedding_http_error(e: GoogleGenerativeAIError) -> None:
    logger.error("Embedding error: %s", e)
    raise HTTPException(
        status_code=500,
        detail="Google Embedding Error: โปรดตรวจสอบ API Key ของคุณ",
    ) from e


# -----------------------------------------------------------
# 1) Indexing: เอา chunks ไปเก็บใน Chroma
# -----------------------------------------------------------

def index_chunks(
    chunks: List[Chunk],
    persist_directory: str = CHROMA_DIR,
    collection_name: str = COLLECTION_NAME,
) -> None:
    if not chunks:
        return

    vectordb = get_vector_store(persist_directory, collection_name)
    texts = [c.content for c in chunks]
    
    # รวม metadata พื้นฐาน + ข้อมูล Chunk
    raw_metadatas = [
        (c.metadata or {}) | {
            "doc_id": c.doc_id,
            "doc_type": c.doc_type,
            "source": c.source,
            "page": c.page,
            "chunk_id": c.id,
        } for c in chunks
    ]
    
    metadatas = [_normalize_metadata(md) for md in raw_metadatas]
    ids = [c.id for c in chunks]

    try:
        logger.info(f"[vector_store] Indexing {len(chunks)} chunks...")
        vectordb.add_texts(texts=texts, metadatas=metadatas, ids=ids)
        try:
            vectordb.persist()
        except Exception: 
            pass
    except GoogleGenerativeAIError as e:
        _raise_embedding_http_error(e)
    except Exception as e:
        logger.exception("[vector_store] Indexing error: %s", e)
        raise HTTPException(status_code=500, detail=f"Indexing error: {e}") from e


# -----------------------------------------------------------
# 2) Search: Pure Retrieval (No AI Logic)
# -----------------------------------------------------------

def _python_filter_documents(raw_docs: List[Document], doc_ids: Optional[List[str]], sources: Optional[List[str]], doc_types: Optional[List[str]]) -> List[Document]:
    """กรองเอกสารด้วย Python (แม่นยำกว่า Chroma filter ในบางกรณี)"""
    filtered = []
    for d in raw_docs:
        md = d.metadata or {}
        if doc_ids and md.get("doc_id") not in doc_ids:
            continue
        if sources and md.get("source") not in sources:
            continue
        if doc_types and md.get("doc_type") not in doc_types:
            continue
        filtered.append(d)
    return filtered


def search_similar(
    query: str,
    k: int = 5,
    persist_directory: str = CHROMA_DIR,
    collection_name: str = COLLECTION_NAME,
    doc_ids: Optional[List[str]] = None,
    sources: Optional[List[str]] = None,
    doc_types: Optional[List[str]] = None,
) -> List[Document]:
    """
    หน้าที่: ดึงข้อมูลจาก DB ให้ได้ตามจำนวน k ที่ขอ (หลังกรองแล้ว)
    * ตัด Re-ranking ออกแล้ว (ย้ายไป rag.py)
    """
    if not query or not query.strip():
        raise HTTPException(status_code=400, detail="Empty query")

    vectordb = get_vector_store(persist_directory, collection_name)

    # Fetch more to allow for filtering (k * 4)
    # แต่ไม่จำเป็นต้องเยอะเท่าตอนมี Re-ranking
    fetch_k = max(k * 4, 20)

    try:
        raw_docs = vectordb.similarity_search(query, k=fetch_k)
    except ChromaInternalError:
        # Retry mechanism for stability
        logger.warning("[vector_store] Chroma InternalError. Retrying...")
        key = _cache_key(persist_directory, collection_name)
        _vectordb_cache.pop(key, None)
        vectordb = get_vector_store(persist_directory, collection_name, force_recreate=True)
        raw_docs = vectordb.similarity_search(query, k=fetch_k)
    except Exception as e:
        logger.exception("[vector_store] Search failed: %s", e)
        raise HTTPException(status_code=500, detail="Vector search failed") from e

    # Filter in Python
    filtered_docs = _python_filter_documents(raw_docs, doc_ids, sources, doc_types)

    # Return exactly k (or fewer)
    result = filtered_docs[:k]
    
    logger.debug(f"[vector_store] Query: '{query}' | Fetched: {len(raw_docs)} | Filtered: {len(filtered_docs)} | Returned: {len(result)}")
    return result