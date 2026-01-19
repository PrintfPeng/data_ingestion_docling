from __future__ import annotations

import os
from typing import List, Dict, Any, Optional

from dotenv import load_dotenv
# [CHANGE] เปลี่ยนจาก OpenAIEmbeddings เป็น HuggingFaceEmbeddings (Local)
# เพื่อแก้ปัญหา Server ไม่รองรับ Embeddings หรือ Chat Model ทำ Embeddings ไม่ได้
from langchain_huggingface import HuggingFaceEmbeddings

# [CHANGE] ใช้โมเดล Embeddings จริงๆ (Multilingual) แทน Chat Model
_EMBEDDING_MODEL_NAME = "intfloat/multilingual-e5-large" 
# หรือถ้าเครื่องช้าใช้: "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

# เก็บ client แบบ singleton
# [CHANGE] อัปเดต Type Hint
_embeddings_client: HuggingFaceEmbeddings | None = None


# =============================================================================
# >>> EMBEDDINGS CONTRACT FIX <<<
# =============================================================================
# DOCUMENTATION & SAFETY GUARANTEE:
# 1. This layer generates vectors ONLY from the 'text' content.
# 2. It DOES NOT and MUST NOT embed, normalize, or alter 'metadata'.
# 3. All metadata (including 'source': 'table', JSON fields, HTML) MUST pass
#    through to the vector store completely intact.
# 4. Any logic affecting table/image metadata must happen in chunking.py, NOT here.
# =============================================================================


def get_embedding_client() -> HuggingFaceEmbeddings:
    """
    คืน client สำหรับสร้าง embedding โดยรัน Local (HuggingFace)
    """
    global _embeddings_client

    if _embeddings_client is None:
        print(f"⏳ Loading Local Embedding Model: {_EMBEDDING_MODEL_NAME} ...")
        # [CHANGE] สร้าง Client แบบ Local HuggingFace
        _embeddings_client = HuggingFaceEmbeddings(
            model_name=_EMBEDDING_MODEL_NAME,
            model_kwargs={'device': 'cpu'}, # เปลี่ยนเป็น 'cuda' ถ้ามี GPU
            encode_kwargs={'normalize_embeddings': True}
        )

    return _embeddings_client


def embed_texts(texts: List[str]) -> List[List[float]]:
    """
    helper สำหรับ embed เป็น batch จาก list ของข้อความ
    NOTE: Functions using this must handle metadata association manually.
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


# =============================================================================
# >>> EMBEDDINGS CONTRACT FIX <<<
# =============================================================================

def embed_with_metadata(items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Safe helper that generates embeddings while GUARANTEEING metadata preservation.
    
    Args:
        items: List of dicts, each must have {"text": str, "metadata": dict}
               (Optional fields in metadata are allowed and preserved)

    Returns:
        List of dicts: [{"embedding": [...], "metadata": {...}}, ...]

    Behavior:
    - Extracts 'text' for embedding generation.
    - Passthroughs 'metadata' blindly (no modification, no stripping).
    - Ensures 1-to-1 mapping between vector and metadata.
    """
    if not items:
        return []

    # 1. Extract texts strictly for vector generation
    # Default to empty string if text missing to prevent crashes, though caller should validate
    texts = [str(item.get("text", "")) for item in items]

    # 2. Generate vectors (Batch operation)
    vectors = embed_texts(texts)

    # 3. Re-attach metadata without alteration
    results = []
    for i, vector in enumerate(vectors):
        original_meta = items[i].get("metadata", {})
        
        # Explicit guarantee: Source metadata (e.g., tables) is never dropped
        results.append({
            "embedding": vector,
            "metadata": original_meta  # Pass by reference is fine here
        })

    return results