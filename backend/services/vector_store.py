from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Dict, Tuple

import logging
from fastapi import HTTPException
from langchain_community.vectorstores import Chroma
from langchain_google_genai._common import GoogleGenerativeAIError

from .chunking import Chunk
from .embeddings import get_embedding_client

# -----------------------------------------------------------
# ตั้งค่าพื้นฐาน
# -----------------------------------------------------------

logger = logging.getLogger(__name__)

CHROMA_DIR = "chroma_db"
COLLECTION_NAME = "documents"

# cache vectordb ตาม (persist_directory, collection_name)
_vectordb_cache: Dict[Tuple[str, str], Chroma] = {}


# -----------------------------------------------------------
# Helper: เตรียม / reuse Chroma vector store
# -----------------------------------------------------------

def get_vector_store(
    persist_directory: str = CHROMA_DIR,
    collection_name: str = COLLECTION_NAME,
) -> Chroma:
    """
    คืน Chroma vector store ที่ผูกกับ Gemini embeddings
    - มี cache ใน process เดียวกันเพื่อไม่ต้องสร้างใหม่ทุกครั้ง
    """
    persist_path = Path(persist_directory)
    persist_path.mkdir(parents=True, exist_ok=True)

    key = (str(persist_path), collection_name)
    if key in _vectordb_cache:
        return _vectordb_cache[key]

    embeddings = get_embedding_client()

    vectordb = Chroma(
        collection_name=collection_name,
        embedding_function=embeddings,
        persist_directory=str(persist_path),
    )

    _vectordb_cache[key] = vectordb
    return vectordb


def _normalize_metadata(md: dict) -> dict:
    """
    Chroma รับได้เฉพาะค่าแบบ str/int/float/bool/None
    อันนี้เลยแปลงพวก list/dict/object ให้กลายเป็น string
    """
    simple: dict = {}
    for k, v in md.items():
        if isinstance(v, (str, int, float, bool)) or v is None:
            simple[k] = v
        else:
            # แปลงของซับซ้อนเป็น string เช่น bbox, columns, ฯลฯ
            simple[k] = str(v)
    return simple


def _raise_embedding_http_error(e: GoogleGenerativeAIError) -> None:
    """
    แปลง error จากฝั่ง embedding ให้เป็น HTTPException ที่อ่านรู้เรื่อง
    ใช้ร่วมกันได้ทั้งตอน index และตอน search
    """
    logger.error("Embedding error: %s", e)
    raise HTTPException(
        status_code=500,
        detail=(
            "Embedding error จากบริการ Gemini: โปรดตรวจสอบ GOOGLE_API_KEY ใน .env "
            "(อาจใช้ไม่ได้ / หมดอายุ / ถูกปิด / ถูกจำกัดสิทธิ์)"
        ),
    ) from e


# -----------------------------------------------------------
# 1) Indexing: เอา chunks ไปเก็บใน Chroma
# -----------------------------------------------------------

def index_chunks(
    chunks: List[Chunk],
    persist_directory: str = CHROMA_DIR,
    collection_name: str = COLLECTION_NAME,
) -> None:
    """
    เอา chunks ทั้งหมดไปเก็บใน Chroma
    - รองรับการเรียกซ้ำ (append entries เพิ่มใน collection เดิม)
    - ใส่ metadata ที่จำเป็นสำหรับ filter: doc_id, doc_type, source, page, chunk_id
    """
    if not chunks:
        logger.info("[vector_store] No chunks to index, skip.")
        return

    vectordb = get_vector_store(
        persist_directory=persist_directory,
        collection_name=collection_name,
    )

    texts = [c.content for c in chunks]

    raw_metadatas = [
        c.metadata
        | {
            "doc_id": c.doc_id,
            "doc_type": c.doc_type,
            "source": c.source,
            "page": c.page,
            "chunk_id": c.id,
        }
        for c in chunks
    ]

    metadatas = [_normalize_metadata(md) for md in raw_metadatas]
    ids = [c.id for c in chunks]

    try:
        logger.info(
            "[vector_store] Indexing %d chunks into Chroma (collection=%s)",
            len(chunks),
            collection_name,
        )
        vectordb.add_texts(
            texts=texts,
            metadatas=metadatas,
            ids=ids,
        )

        # Chroma 0.4+ จะ persist อัตโนมัติอยู่แล้ว แต่เรียกซ้ำก็ไม่เป็นไร
        try:
            vectordb.persist()
        except Exception as pe:  # noqa: BLE001
            # ไม่ถือว่า fatal แค่เตือน
            logger.warning("[vector_store] Chroma.persist() warning: %r", pe)

    except GoogleGenerativeAIError as e:
        _raise_embedding_http_error(e)


# -----------------------------------------------------------
# 2) Search: similarity search + filter
# -----------------------------------------------------------

def search_similar(
    query: str,
    k: int = 5,
    persist_directory: str = CHROMA_DIR,
    collection_name: str = COLLECTION_NAME,
    doc_ids: Optional[List[str]] = None,
    sources: Optional[List[str]] = None,
):
    """
    search จาก Chroma พร้อม filter ตาม doc_ids / source ได้
    ใช้ syntax filter ของ Chroma:

    - ถ้ามีเงื่อนไขเดียว → {"doc_id": {"$in": [...]}}
    - ถ้ามีหลายเงื่อนไข → {"$and": [ {...}, {...} ]}
    """

    vectordb = get_vector_store(
        persist_directory=persist_directory,
        collection_name=collection_name,
    )

    conditions: List[dict] = []

    if doc_ids:
        conditions.append({"doc_id": {"$in": doc_ids}})

    if sources:
        conditions.append({"source": {"$in": sources}})

    if len(conditions) == 0:
        filter_dict = None
    elif len(conditions) == 1:
        filter_dict = conditions[0]
    else:
        filter_dict = {"$and": conditions}

    try:
        if filter_dict:
            logger.debug(
                "[vector_store] similarity_search: q=%r, k=%d, filter=%s",
                query,
                k,
                filter_dict,
            )
            docs = vectordb.similarity_search(
                query,
                k=k,
                filter=filter_dict,
            )
        else:
            logger.debug(
                "[vector_store] similarity_search: q=%r, k=%d (no filter)",
                query,
                k,
            )
            docs = vectordb.similarity_search(query, k=k)

        return docs

    except GoogleGenerativeAIError as e:
        _raise_embedding_http_error(e)
