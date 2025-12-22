from __future__ import annotations
import re
from typing import Any, Dict, List, Literal, Optional
from pydantic import BaseModel, Field
from docling.document_converter import DocumentConverter
from docling_core.transforms.chunker.hierarchical_chunker import HierarchicalChunker

# --- Configuration ---
_TARGET_CHARS = 1000
_MAX_CHUNK_SIZE = 1500
_CHUNK_OVERLAP = 200

class Chunk(BaseModel):
    id: str
    doc_id: str
    doc_type: str
    source: Literal["text", "table", "image"]
    page: Optional[int] = None
    content: str
    metadata: Dict[str, Any] = Field(default_factory=dict)

# -------------------------------------------------------------------
# Helper: Text Normalization (แก้ปัญหาภาษาไทยตรงนี้)
# -------------------------------------------------------------------

def fix_thai_vowels(text: str) -> str:
    """
    ฟังก์ชันซ่อมแซมสระและวรรณยุกต์ภาษาไทยที่มักผิดพลาดจาก OCR
    """
    if not text:
        return ""
    
    # 1. แก้สระอำที่แยกร่าง (เช่น ํ า -> ำ)
    text = text.replace("ํ า", "ำ").replace("ํา", "ำ")
    
    # 2. แก้สระที่ชอบลอยหรือจมผิดปกติ (Cleanup Zero-width spaces or weird artifacts)
    # ลบช่องว่างที่อาจคั่นระหว่างพยัญชนะกับสระ (เช่น ก า ร -> การ) เฉพาะเคสที่มั่นใจ
    # (Regex นี้จะช่วยดึงสระบน/ล่างที่ลอยห่างกลับมาติดพยัญชนะ)
    text = re.sub(r'([ก-ฮ])\s+([ะ-ูเ-ไโ็่-๋์])', r'\1\2', text)
    
    # 3. ลบตัวอักษรขยะที่พบบ่อยใน OCR ภาษาไทย
    text = text.replace("", "") 
    
    return text

def _normalize_whitespace(text: str) -> str:
    if not text:
        return ""
    
    # ซ่อมภาษาไทยก่อน
    text = fix_thai_vowels(text)
    
    # ยุบ space แต่เก็บ newline ไว้
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()

# -------------------------------------------------------------------
# Main Chunker Class
# -------------------------------------------------------------------

class MarkdownChunker:
    def __init__(self):
        self.chunker = HierarchicalChunker()

    def create_chunks(self, doc) -> List[Dict[str, Any]]:
        """
        ตัดแบ่ง Markdown จาก DoclingDocument เป็น Chunks
        """
        # ใช้ Docling core chunker ตัดแบ่งตามโครงสร้าง (Header, Paragraph)
        chunks = self.chunker.chunk(doc)
        
        enriched_chunks = []
        for i, chunk in enumerate(chunks):
            # Serialize เนื้อหาของ chunk นั้นกลับมาเป็น text
            chunk_text = self.chunker.serialize(chunk)
            
            # Normalize ข้อความอีกครั้งเพื่อความชัวร์
            chunk_text = _normalize_whitespace(chunk_text)
            
            if not chunk_text.strip():
                continue

            # ดึงหัวข้อ (Heading) เพื่อทำ Semantic Context
            heading = self._get_main_heading(chunk)
            
            # สร้าง Metadata
            metadata = {
                "chunk_id": i,
                "heading": heading,
                "page": self._get_page_number(chunk),
                "source": "docling_parser"
            }
            
            enriched_chunks.append({
                "content": chunk_text,
                "metadata": metadata
            })
            
        return enriched_chunks

    def _get_main_heading(self, chunk):
        if hasattr(chunk, 'meta') and hasattr(chunk.meta, 'headings') and chunk.meta.headings:
            return " > ".join(chunk.meta.headings)
        return "General Content"

    def _get_page_number(self, chunk):
        # พยายามดึงเลขหน้าจาก provenance item แรก
        if hasattr(chunk, 'meta') and hasattr(chunk.meta, 'doc_items') and chunk.meta.doc_items:
            return chunk.meta.doc_items[0].prov[0].page_no
        return 1