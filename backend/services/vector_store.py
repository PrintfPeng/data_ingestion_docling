from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Dict, Tuple
import logging
import warnings
import re

from fastapi import HTTPException
from langchain_community.vectorstores import Chroma
from langchain_google_genai._common import GoogleGenerativeAIError
from langchain_core.documents import Document

from .chunking import Chunk
from .embeddings import get_embedding_client


# from backend.services.vector_store import get_collection_info
# info = get_collection_info()
# print(info)
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
# [NEW] Sanitize Document ID (ให้ตรงกับ rag.py)
# -----------------------------------------------------------
def sanitize_doc_id(doc_id: str) -> str:
    """
    Sanitize document ID ให้ตรงกับรูปแบบที่เก็บใน DB
    - Lowercase
    - Replace spaces with underscores
    - Remove special characters except underscore
    
    Example: "Sample QNA" -> "sample_qna"
    """
    if not doc_id:
        return ""
    
    doc_id = str(doc_id).lower().strip()
    doc_id = re.sub(r'\s+', '_', doc_id)
    doc_id = re.sub(r'[^a-z0-9_]', '', doc_id)
    
    return doc_id


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
    
    # [FIX] Sanitize doc_id ก่อนเก็บ
    raw_metadatas = [
        (c.metadata or {}) | {
            "doc_id": sanitize_doc_id(c.doc_id),  # [CRITICAL FIX]
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
        
        # [DEBUG] แสดง doc_id ที่เก็บลงไป
        unique_doc_ids = set(md["doc_id"] for md in raw_metadatas)
        logger.info(f"[vector_store] Indexed doc_ids: {unique_doc_ids}")
        
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
# 2) Search: Pure Retrieval (COMPLETELY REWRITTEN)
# -----------------------------------------------------------

def _python_filter_documents(
    raw_docs: List[Document], 
    doc_ids: Optional[List[str]], 
    sources: Optional[List[str]], 
    doc_types: Optional[List[str]]
) -> List[Document]:
    """
    [IMPROVED] กรองเอกสารด้วย Python พร้อม Sanitization
    """
    filtered = []
    
    # [FIX] Sanitize doc_ids ที่ใช้กรอง
    sanitized_doc_ids = None
    if doc_ids:
        sanitized_doc_ids = set(sanitize_doc_id(d) for d in doc_ids)
    
    for d in raw_docs:
        md = d.metadata or {}
        
        # [DEBUG] Log metadata ของ document แรก
        if not filtered:
            logger.debug(f"[vector_store] Sample metadata: {md}")
        
        # Check doc_ids (WITH SANITIZATION)
        if sanitized_doc_ids:
            found_id = md.get("doc_id")
            if not found_id:
                continue
            
            # [FIX] Sanitize ก่อนเปรียบเทียบ
            normalized_found_id = sanitize_doc_id(str(found_id))
            if normalized_found_id not in sanitized_doc_ids:
                continue
                
        # Check sources
        if sources:
            doc_source = md.get("source")
            if not doc_source or str(doc_source) not in sources:
                continue
            
        # Check doc_types
        if doc_types:
            doc_type = md.get("doc_type")
            if not doc_type or str(doc_type) not in doc_types:
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
    [COMPLETELY REWRITTEN] ระบบค้นหาแบบ Robust with Smart Fallback
    
    Strategy:
    1. Try Native Chroma Filter (Fast)
    2. If fails -> Python-side Filter (Safe)
    3. Log everything for debugging
    """
    if not query or not query.strip():
        raise HTTPException(status_code=400, detail="Empty query")

    vectordb = get_vector_store(persist_directory, collection_name)

    # [FIX] Sanitize doc_ids ก่อนสร้าง filter
    sanitized_doc_ids = None
    if doc_ids:
        sanitized_doc_ids = [sanitize_doc_id(d) for d in doc_ids if d]
        logger.info(f"[vector_store] Original doc_ids: {doc_ids} -> Sanitized: {sanitized_doc_ids}")

    # --- 1. สร้างเงื่อนไขการกรอง (SIMPLIFIED) ---
    where_filter = {}
    
    # [FIX] ใช้ Simple Equality แทน $in สำหรับ Single Value
    if sanitized_doc_ids:
        if len(sanitized_doc_ids) == 1:
            where_filter["doc_id"] = sanitized_doc_ids[0]
        else:
            # [FIX] บาง Chroma version ไม่รองรับ $in ให้ลอง OR แทน
            # แต่ OR ก็ซับซ้อน ดังนั้นถ้ามี multiple IDs ให้ใช้ Python Filter แทน
            where_filter = None  # Force Python Filter
            
    if sources:
        if len(sources) == 1:
            if where_filter is not None:
                where_filter["source"] = sources[0]
        else:
            where_filter = None  # Force Python Filter
            
    if doc_types:
        if len(doc_types) == 1:
            if where_filter is not None:
                where_filter["doc_type"] = doc_types[0]
        else:
            where_filter = None

    # --- 2. Smart Search Strategy ---
    try:
        # [NEW] Strategy: ถ้า Filter ซับซ้อน (multiple values) ให้ข้าม Native Filter ไปเลย
        use_native_filter = where_filter is not None and where_filter != {}
        
        if use_native_filter:
            logger.info(f"[vector_store] Using NATIVE filter: {where_filter}")
            results = vectordb.similarity_search(query, k=k, filter=where_filter)
            
            # Check if native filter worked
            if not results:
                logger.warning(f"[vector_store] Native filter returned 0 results. Switching to Python filter.")
                use_native_filter = False  # Trigger fallback
        
        # Fallback to Python Filter
        if not use_native_filter:
            logger.info(f"[vector_store] Using PYTHON filter for: doc_ids={sanitized_doc_ids}, sources={sources}, doc_types={doc_types}")
            
            # [FIX] ดึงมา k*10 แทน k*5 เพื่อเพิ่มโอกาสเจอ
            fetch_size = max(k * 10, 50)  # อย่างน้อย 50 ตัว
            raw_docs = vectordb.similarity_search(query, k=fetch_size)
            
            logger.info(f"[vector_store] Fetched {len(raw_docs)} raw documents for Python filtering")
            
            # [DEBUG] แสดง doc_ids ที่ดึงมาได้
            if raw_docs:
                found_doc_ids = set(d.metadata.get("doc_id") for d in raw_docs if d.metadata)
                logger.info(f"[vector_store] Available doc_ids in fetched results: {found_doc_ids}")
            
            results = _python_filter_documents(raw_docs, doc_ids, sources, doc_types)[:k]
        
        logger.info(f"[vector_store] Search query='{query[:50]}...' returned {len(results)} results")
        
        # [DEBUG] แสดง doc_ids ของผลลัพธ์
        if results:
            result_doc_ids = [d.metadata.get("doc_id") for d in results if d.metadata]
            logger.info(f"[vector_store] Result doc_ids: {result_doc_ids}")
        else:
            logger.warning(f"[vector_store] ⚠️ No results found! This might indicate a problem.")
        
        return results

    except ChromaInternalError:
        logger.warning("[vector_store] Chroma InternalError. Retrying...")
        key = _cache_key(persist_directory, collection_name)
        _vectordb_cache.pop(key, None)
        vectordb = get_vector_store(persist_directory, collection_name, force_recreate=True)
        
        # Retry with Python Filter only (safest)
        raw_docs = vectordb.similarity_search(query, k=k*10)
        results = _python_filter_documents(raw_docs, doc_ids, sources, doc_types)[:k]
        return results
        
    except Exception as e:
        logger.exception("[vector_store] Search failed: %s", e)
        
        # Ultimate Fallback
        try:
            logger.warning("[vector_store] Exception occurred, using ultimate fallback (Python filter)")
            raw_docs = vectordb.similarity_search(query, k=k*10)
            return _python_filter_documents(raw_docs, doc_ids, sources, doc_types)[:k]
        except Exception as final_error:
            logger.exception("[vector_store] Ultimate fallback also failed: %s", final_error)
            raise HTTPException(status_code=500, detail="Vector search completely failed") from final_error


# -----------------------------------------------------------
# [NEW] Debug Helper: Inspect Collection
# -----------------------------------------------------------
def get_collection_info(
    persist_directory: str = CHROMA_DIR,
    collection_name: str = COLLECTION_NAME,
) -> Dict:
    """
    [NEW] Debug function: แสดงข้อมูลใน Collection
    """
    try:
        vectordb = get_vector_store(persist_directory, collection_name)
        
        # Get sample documents
        sample_docs = vectordb.similarity_search("test", k=5)
        
        # Extract unique doc_ids
        doc_ids = set()
        sources = set()
        doc_types = set()
        
        for doc in sample_docs:
            md = doc.metadata or {}
            if md.get("doc_id"):
                doc_ids.add(md.get("doc_id"))
            if md.get("source"):
                sources.add(md.get("source"))
            if md.get("doc_type"):
                doc_types.add(md.get("doc_type"))
        
        return {
            "collection_name": collection_name,
            "sample_count": len(sample_docs),
            "unique_doc_ids": list(doc_ids),
            "unique_sources": list(sources),
            "unique_doc_types": list(doc_types),
            "sample_metadata": [doc.metadata for doc in sample_docs[:3]]
        }
    except Exception as e:
        logger.exception("[vector_store] Failed to get collection info: %s", e)
        return {"error": str(e)}