# backend/services/chunking.py

from __future__ import annotations
import re
import uuid
from typing import Any, Dict, List, Literal, Optional
from pathlib import Path
from pydantic import BaseModel, Field

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
        return self.content

# -------------------------------------------------------------------
# Helper: Text Normalization
# -------------------------------------------------------------------
def fix_thai_vowels(text: str) -> str:
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
# Converter Functions
# -------------------------------------------------------------------

def text_items_to_chunks(bundle: Any) -> List[Chunk]:
    chunks = []
    items = getattr(bundle, "texts", []) or []
    
    meta_obj = getattr(bundle, "metadata", None)
    doc_id = getattr(meta_obj, "doc_id", "unknown") if meta_obj else "unknown"
    doc_type = getattr(meta_obj, "doc_type", "generic") if meta_obj else "generic"

    current_chunk_text = ""
    current_meta = {}
    
    for item in items:
        # ดึง text
        if isinstance(item, dict): raw_text = item.get("content", "")
        else: raw_text = getattr(item, "content", "")
        
        clean_text = _normalize_whitespace(raw_text)
        if not clean_text: continue

        if len(current_chunk_text) + len(clean_text) < _TARGET_CHARS:
             current_chunk_text += "\n" + clean_text
             if not current_meta: 
                 if isinstance(item, dict): current_meta = item
                 else: current_meta = item.model_dump() if hasattr(item, "model_dump") else item.__dict__
        else:
             chunks.append(Chunk(
                id=str(uuid.uuid4()),
                doc_id=doc_id,
                doc_type=doc_type,
                source="text",
                page=current_meta.get("page", 1),
                content=current_chunk_text.strip(),
                metadata=current_meta
             ))
             current_chunk_text = clean_text
             if isinstance(item, dict): current_meta = item
             else: current_meta = item.model_dump() if hasattr(item, "model_dump") else item.__dict__

    if current_chunk_text:
        chunks.append(Chunk(
            id=str(uuid.uuid4()),
            doc_id=doc_id,
            doc_type=doc_type,
            source="text",
            page=current_meta.get("page", 1),
            content=current_chunk_text.strip(),
            metadata=current_meta
        ))
        
    return chunks

def table_items_to_chunks(bundle: Any) -> List[Chunk]:
    chunks = []
    items = getattr(bundle, "tables", []) or []
    
    meta_obj = getattr(bundle, "metadata", None)
    doc_id = getattr(meta_obj, "doc_id", "unknown") if meta_obj else "unknown"
    doc_type = getattr(meta_obj, "doc_type", "generic") if meta_obj else "generic"
    
    for item in items:
        if isinstance(item, dict):
            content = item.get("markdown_content") or item.get("content") or ""
            page = item.get("page")
            if "html_content" not in item and "content" in item:
                 item["html_content"] = item["content"]
            meta = item
        else:
            rows = getattr(item, "rows", [])
            cols = getattr(item, "columns", [])
            extra = getattr(item, "extra", {}) or {}
            content = extra.get("markdown_content", "")
            
            if not content:
                 header = "| " + " | ".join(cols) + " |"
                 sep = "| " + " | ".join(["---"] * len(cols)) + " |"
                 body_rows = ["| " + " | ".join(r) + " |" for r in rows]
                 content = f"{header}\n{sep}\n" + "\n".join(body_rows)

            page = getattr(item, "page", None)
            meta = item.model_dump() if hasattr(item, "model_dump") else item.__dict__

        if not content.strip(): continue

        chunks.append(Chunk(
            id=str(uuid.uuid4()),
            doc_id=doc_id,
            doc_type=doc_type,
            source="table",
            page=page,
            content=content,
            metadata=meta
        ))
    return chunks

def image_items_to_chunks(bundle: Any) -> List[Chunk]:
    chunks = []
    items = getattr(bundle, "images", []) or []
    
    meta_obj = getattr(bundle, "metadata", None)
    doc_id = getattr(meta_obj, "doc_id", "unknown") if meta_obj else "unknown"
    doc_type = getattr(meta_obj, "doc_type", "generic") if meta_obj else "generic"
    
    for item in items:
        if isinstance(item, dict):
            caption = item.get("caption", "")
            file_path = item.get("file_path", "")
            page = item.get("page")
            meta = item
        else:
            caption = getattr(item, "caption", "")
            file_path = getattr(item, "file_path", "")
            page = getattr(item, "page", None)
            meta = item.model_dump() if hasattr(item, "model_dump") else item.__dict__

        if not caption: caption = f"Image from page {page}"

        # [NEW LOGIC] สร้าง Content ที่มี Markdown Link
        # รูปแบบ: ![ชื่อรูป](/ingested/doc_id/images/xxx.png)
        # AI จะเห็น Text นี้และสามารถเอาไปตอบได้
        
        img_filename = Path(file_path).name
        if img_filename:
            web_path = f"/ingested/{doc_id}/images/{img_filename}"
            # รวม Link ไว้ใน Content เพื่อให้ RAG Index ไปด้วย
            full_content = f"![{img_filename}]({web_path})\n\nคำบรรยายภาพ: {caption}"
        else:
            full_content = caption
        
        chunks.append(Chunk(
            id=str(uuid.uuid4()),
            doc_id=doc_id,
            doc_type=doc_type,
            source="image",
            page=page,
            content=full_content, # ใช้ Content แบบใหม่
            metadata=meta
        ))
    return chunks