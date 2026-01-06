from __future__ import annotations
import re
import uuid
from typing import Any, Dict, List, Literal, Optional, Union
from pydantic import BaseModel, Field
from docling.document_converter import DocumentConverter
from docling_core.transforms.chunker.hierarchical_chunker import HierarchicalChunker

# --- Configuration ---
_TARGET_CHARS = 1000
_MAX_CHUNK_SIZE = 1500
_CHUNK_OVERLAP = 200

# -------------------------------------------------------------------
# Model: Chunk
# -------------------------------------------------------------------
class Chunk(BaseModel):
    id: str
    doc_id: str
    doc_type: str
    source: Literal["text", "table", "image"]
    page: Optional[int] = None
    content: str
    metadata: Dict[str, Any] = Field(default_factory=dict)

    @property
    def page_content(self) -> str:
        """Helper for LangChain compatibility"""
        return self.content

# -------------------------------------------------------------------
# Helper: Text Normalization
# -------------------------------------------------------------------
def fix_thai_vowels(text: str) -> str:
    """ซ่อมสระภาษาไทยจาก OCR"""
    if not text: return ""
    text = text.replace("ํ า", "ำ").replace("ํา", "ำ")
    text = re.sub(r'([ก-ฮ])\s+([ะ-ูเ-ไโ็่-๋์])', r'\1\2', text)
    return text

def _normalize_whitespace(text: str) -> str:
    if not text: return ""
    text = fix_thai_vowels(text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()

# -------------------------------------------------------------------
# Converter Functions (แก้ไข: รองรับ Pydantic Object)
# -------------------------------------------------------------------

def text_items_to_chunks(bundle: Any) -> List[Chunk]:
    """แปลงรายการ Text เป็น Chunk"""
    chunks = []
    # แก้การเรียกใช้ข้อมูลจาก bundle.get() เป็น bundle.attribute
    items = getattr(bundle, "text_items", []) or []
    doc_id = getattr(bundle, "doc_id", "unknown")
    doc_type = getattr(bundle, "doc_type", "generic")
    
    for item in items:
        raw_text = item.get("content", "")
        clean_text = _normalize_whitespace(raw_text)
        if not clean_text:
            continue
            
        chunks.append(Chunk(
            id=str(uuid.uuid4()),
            doc_id=doc_id,
            doc_type=doc_type,
            source="text",
            page=item.get("page"),
            content=clean_text,
            metadata=item
        ))
    return chunks

def table_items_to_chunks(bundle: Any) -> List[Chunk]:
    """แปลงรายการ Table เป็น Chunk"""
    chunks = []
    items = getattr(bundle, "table_items", []) or []
    doc_id = getattr(bundle, "doc_id", "unknown")
    doc_type = getattr(bundle, "doc_type", "generic")
    
    for item in items:
        # ใช้ Markdown เป็นตัวแทนตารางในการ Search
        content = item.get("markdown_content") or item.get("content") or ""
        if not content.strip():
            continue
            
        # สำคัญ: เก็บ HTML ไว้ใน metadata เพื่อให้ RAG ดึงไปแสดงผล
        if "html_content" not in item and "content" in item:
             item["html_content"] = item["content"]

        chunks.append(Chunk(
            id=str(uuid.uuid4()),
            doc_id=doc_id,
            doc_type=doc_type,
            source="table",
            page=item.get("page"),
            content=content,
            metadata=item
        ))
    return chunks

def image_items_to_chunks(bundle: Any) -> List[Chunk]:
    """แปลงรายการ Image เป็น Chunk"""
    chunks = []
    items = getattr(bundle, "image_items", []) or []
    doc_id = getattr(bundle, "doc_id", "unknown")
    doc_type = getattr(bundle, "doc_type", "generic")
    
    for item in items:
        # ใช้ Caption เป็นตัวแทนในการ Search
        caption = item.get("caption", "")
        if not caption:
            caption = f"Image from page {item.get('page')}"
        
        chunks.append(Chunk(
            id=str(uuid.uuid4()),
            doc_id=doc_id,
            doc_type=doc_type,
            source="image",
            page=item.get("page"),
            content=caption,
            metadata=item
        ))
    return chunks

# -------------------------------------------------------------------
# Main Chunker Class
# -------------------------------------------------------------------
class MarkdownChunker:
    def __init__(self):
        self.chunker = HierarchicalChunker()

    def create_chunks(self, doc) -> List[Dict[str, Any]]:
        chunks = self.chunker.chunk(doc)
        enriched_chunks = []
        for i, chunk in enumerate(chunks):
            chunk_text = self.chunker.serialize(chunk)
            chunk_text = _normalize_whitespace(chunk_text)
            if not chunk_text.strip(): continue

            heading = self._get_main_heading(chunk)
            metadata = {
                "chunk_id": i,
                "heading": heading,
                "page": self._get_page_number(chunk),
                "source": "docling_parser"
            }
            enriched_chunks.append({"content": chunk_text, "metadata": metadata})
        return enriched_chunks

    def _get_main_heading(self, chunk):
        if hasattr(chunk, 'meta') and hasattr(chunk.meta, 'headings') and chunk.meta.headings:
            return " > ".join(chunk.meta.headings)
        return "General Content"

    def _get_page_number(self, chunk):
        if hasattr(chunk, 'meta') and hasattr(chunk.meta, 'doc_items') and chunk.meta.doc_items:
            return chunk.meta.doc_items[0].prov[0].page_no
        return 1