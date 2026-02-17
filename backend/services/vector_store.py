# backend/services/vector_store.py

from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Dict, Tuple, Any
import logging
import warnings
import re
import gc

from fastapi import HTTPException
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document

from .chunking import Chunk
from .embeddings import get_embedding_client


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

_vectordb_cache: Dict[Tuple[str, str], Chroma] = {}


# -----------------------------------------------------------
# Sanitize Document ID
# -----------------------------------------------------------
def sanitize_doc_id(doc_id: str) -> str:
    if not doc_id:
        return ""
    doc_id = doc_id.lower().strip()
    doc_id = re.sub(r'\s+', '_', doc_id)
    doc_id = re.sub(r'[^a-z0-9_\u0E00-\u0E7F-]', '', doc_id) 
    return doc_id


# -----------------------------------------------------------
# Helper: Chroma DB Management
# -----------------------------------------------------------

def _cache_key(persist_directory: str, collection_name: str) -> Tuple[str, str]:
    return (str(Path(persist_directory).resolve()), collection_name)

def _normalize_metadata(md: dict) -> dict:
    simple: dict = {}
    for k, v in (md or {}).items():
        if isinstance(v, (str, int, float, bool)) or v is None:
            simple[k] = v
        elif isinstance(v, list):
             simple[k] = ",".join([str(x) for x in v])
        else:
            try:
                simple[k] = str(v)
            except Exception:
                simple[k] = repr(v)
    return simple

def get_vector_store(
    persist_directory: str = CHROMA_DIR,
    collection_name: str = COLLECTION_NAME,
    force_recreate: bool = False,
    reload: bool = False,
) -> Chroma:
    should_reload = force_recreate or reload

    persist_path = Path(persist_directory)
    persist_path.mkdir(parents=True, exist_ok=True)
    key = _cache_key(persist_directory, collection_name)

    if should_reload and key in _vectordb_cache:
        logger.info(f"[vector_store] Forcing reload of ChromaDB client for {key}")
        del _vectordb_cache[key]
        gc.collect()

    if not force_recreate and key in _vectordb_cache:
        return _vectordb_cache[key]

    embeddings = get_embedding_client()

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            vectordb = Chroma(
                collection_name=collection_name,
                embedding_function=embeddings,
                persist_directory=str(persist_path),
            )
    except Exception as e:
        logger.exception("[vector_store] Failed to init Chroma: %s", e)
        gc.collect()
        try:
            vectordb = Chroma(
                collection_name=collection_name,
                embedding_function=embeddings,
                persist_directory=str(persist_path),
            )
        except Exception:
            raise HTTPException(
                status_code=500,
                detail="ไม่สามารถเชื่อมต่อ Vector DB ได้ โปรดตรวจสอบการติดตั้ง"
            ) from e

    _vectordb_cache[key] = vectordb
    return vectordb

def reset_vector_store_cache():
    global _vectordb_cache
    if _vectordb_cache:
        logger.info(f"[vector_store] 🧹 Force clearing cache: {len(_vectordb_cache)} entries...")
        _vectordb_cache.clear()
    
    try:
        gc.collect()
        logger.info("[vector_store] 🗑️ Garbage collection done.")
    except Exception as e:
        logger.error(f"[vector_store] ❌ GC Error: {e}")


# -----------------------------------------------------------
# 1) Indexing
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
    
    raw_metadatas = [
        (c.metadata or {}) | {
            "doc_id": sanitize_doc_id(c.doc_id),
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
    except Exception as e:
        logger.exception("[vector_store] Indexing error: %s", e)
        raise HTTPException(status_code=500, detail=f"Indexing error: {e}") from e


# -----------------------------------------------------------
# 2) Search (Fix: Variable Shadowing)
# -----------------------------------------------------------

def _python_filter_documents(
    raw_docs: List[Document], 
    doc_ids: Optional[List[str]], 
    sources: Optional[List[str]], 
    doc_types: Optional[List[str]],
    custom_filter: Optional[Dict[str, Any]] = None
) -> List[Document]:
    filtered = []
    
    sanitized_doc_ids = None
    if doc_ids:
        sanitized_doc_ids = set(sanitize_doc_id(d) for d in doc_ids)
    
    for d in raw_docs:
        md = d.metadata or {}
        
        # 1. Filter by doc_id
        if sanitized_doc_ids:
            found_id = md.get("doc_id")
            if not found_id or sanitize_doc_id(str(found_id)) not in sanitized_doc_ids:
                continue
                
        # 2. Filter by source
        if sources:
            if str(md.get("source")) not in sources:
                continue
            
        # 3. Filter by doc_type
        if doc_types:
            if str(md.get("doc_type")) not in doc_types:
                continue

        # 4. Custom Filter
        if custom_filter:
            match = True
            for key, val in custom_filter.items():
                if str(md.get(key, "")) != str(val):
                    match = False
                    break
            if not match:
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
    where_filter: Optional[Dict[str, Any]] = None,
) -> List[Document]:
    
    if not query or not query.strip():
        raise HTTPException(status_code=400, detail="Empty query")

    vectordb = get_vector_store(persist_directory, collection_name)

    sanitized_doc_ids = None
    if doc_ids:
        sanitized_doc_ids = [sanitize_doc_id(d) for d in doc_ids if d]

    # --- 1. Construct Chroma Filter ---
    conditions = []

    if sanitized_doc_ids:
        if len(sanitized_doc_ids) == 1:
            conditions.append({"doc_id": sanitized_doc_ids[0]})

    if sources:
        if len(sources) == 1:
            conditions.append({"source": sources[0]})

    if doc_types:
        if len(doc_types) == 1:
            conditions.append({"doc_type": doc_types[0]})

    # [FIX] เปลี่ยนชื่อตัวแปรลูปจาก k, v เป็น key, val เพื่อไม่ให้ทับ argument 'k'
    if where_filter:
        for key, val in where_filter.items():
            conditions.append({key: val})

    final_where = {}
    if len(conditions) == 1:
        final_where = conditions[0]
    elif len(conditions) > 1:
        final_where = {"$and": conditions}
    else:
        final_where = None

    # --- 2. Smart Search Strategy ---
    try:
        use_native_filter = (final_where is not None)
        
        if use_native_filter:
            logger.info(f"[vector_store] Using NATIVE filter: {final_where}")
            # ตรงนี้ 'k' จะยังเป็น int เพราะไม่ถูก loop ข้างบนทับแล้ว
            results = vectordb.similarity_search(query, k=k, filter=final_where)
            
            if not results:
                logger.warning(f"[vector_store] Native filter returned 0 results. Fallback to Python filter.")
                use_native_filter = False 
        
        if not use_native_filter:
            logger.info(f"[vector_store] Using PYTHON filter fallback...")
            fetch_size = max(k * 15, 100)
            
            raw_docs = vectordb.similarity_search(query, k=fetch_size)
            results = _python_filter_documents(raw_docs, doc_ids, sources, doc_types, custom_filter=where_filter)[:k]
        
        logger.info(f"[vector_store] Search returned {len(results)} results")
        return results

    except Exception as e:
        error_msg = str(e)
        logger.warning(f"[vector_store] Search Exception: {error_msg}")

        is_db_corruption = (
            "Nothing found on disk" in error_msg 
            or "InternalError" in error_msg 
            or "sqlite" in error_msg.lower()
            or "Error finding id" in error_msg 
        )

        if is_db_corruption or isinstance(e, ChromaInternalError):
            logger.warning("[vector_store] 🚨 DB Corruption detected. Reloading...")
            vectordb = get_vector_store(persist_directory, collection_name, reload=True)
            try:
                raw_docs = vectordb.similarity_search(query, k=k*15)
                results = _python_filter_documents(raw_docs, doc_ids, sources, doc_types, custom_filter=where_filter)[:k]
                return results
            except Exception:
                return []
        
        raise e

def get_collection_info(
    persist_directory: str = CHROMA_DIR,
    collection_name: str = COLLECTION_NAME,
) -> Dict:
    try:
        vectordb = get_vector_store(persist_directory, collection_name, reload=True)
        return {"count": vectordb._collection.count()}
    except Exception as e:
        return {"error": str(e)}