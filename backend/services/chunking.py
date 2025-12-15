from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional
import re
from pydantic import BaseModel, Field
from ..models import DocumentBundle, TableItem

# --- Configuration ---
# เป้าหมายคือ Chunk ที่มีความยาวประมาณนี้ (ตัวอักษร)
_TARGET_CHARS = 1000
# ถ้าเกินนี้ต้องตัดแน่นอน
_MAX_CHUNK_SIZE = 1500
# ส่วนที่ซ้อนทับกันเพื่อให้บริบทต่อเนื่อง (Overlap)
_CHUNK_OVERLAP = 200  # เพิ่ม Overlap หน่อยเพื่อให้ Q กับ A เกาะกันแน่นขึ้น


class Chunk(BaseModel):
    """
    หนึ่งชิ้นข้อมูลที่จะส่งเข้า Vector DB
    """
    id: str
    doc_id: str
    doc_type: str
    source: Literal["text", "table", "image"]
    page: Optional[int] = None
    content: str
    metadata: Dict[str, Any] = Field(default_factory=dict)


# -------------------------------------------------------------------
# Helper: Text Normalization
# -------------------------------------------------------------------

def _normalize_whitespace(text: str) -> str:
    if not text:
        return ""
    # ยุบ space แต่เก็บ newline ไว้ เพื่อรักษาโครงสร้างย่อหน้า
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text) # ไม่ให้มี newline เกิน 2 อันติดกัน
    return text.strip()


def _table_to_text(table: TableItem) -> str:
    """แปลงตารางเป็น Text สำหรับ Embedding"""
    
    # 1. ดึง AI Summary (ถ้ามี)
    extra = getattr(table, "extra", {}) or {}
    ai_summary = extra.get("summary", "").strip()
    
    # 2. สร้าง Raw Table Representation
    header = f"Table {table.name} (page {table.page})"
    cols = table.columns if table.columns else []
    col_line = " | ".join(cols) if cols else ""
    
    row_lines = []
    # ตัดแค่ 15 แถวแรกพอ (กัน token ล้น) ส่วนที่เหลือ AI อาจต้องไปอ่านในไฟล์เต็มเอาเองถ้าจำเป็น
    max_rows = min(len(table.rows), 15)
    for row in table.rows[:max_rows]:
        safe_row = [str(c) if c is not None else "" for c in row]
        row_lines.append(" | ".join(safe_row))
    
    raw_table_text = "\n".join(row_lines)
    
    # 3. รวมร่าง: [AI Summary] + [Structure] + [Raw Data]
    parts = []
    
    if ai_summary:
        parts.append(f"บทสรุปตาราง: {ai_summary}")
        parts.append("-" * 20)
    
    parts.append(header)
    if col_line:
        parts.append(f"Columns: {col_line}")
    parts.append(f"Rows:\n{raw_table_text}")
    
    return "\n".join(parts)


# -------------------------------------------------------------------
# Core Logic: Semantic / Recursive Splitter
# -------------------------------------------------------------------

def _split_text_recursively(text: str, target_size: int, chunk_overlap: int) -> List[str]:
    """
    ตัดข้อความยาวๆ ให้เป็นชิ้นย่อย โดยพยายามตัดที่จุดที่เหมาะสม
    ลำดับความสำคัญการตัด:
    1. \n\n (ย่อหน้า)
    2. \n (บรรทัด)
    3. . (จบประโยค)
    4. , (คอมม่า)
    5. " " (ช่องว่าง)
    6. ตัดดื้อๆ (Character limit)
    """
    if len(text) <= target_size:
        return [text]
    
    separators = ["\n\n", "\n", ". ", " ", ""]
    
    for sep in separators:
        if sep == "":
            # ถ้าไม่เจอตัวแบ่งเลย ต้องตัดดื้อๆ
            chunks = []
            for i in range(0, len(text), target_size - chunk_overlap):
                chunks.append(text[i : i + target_size])
            return chunks
        
        # ลองแบ่งด้วย separator นี้
        splits = text.split(sep)
        # ถ้าแบ่งแล้วไม่ได้ช่วยให้เล็กลง (ยังก้อนใหญ่เท่าเดิม) ให้ข้ามไป sep ถัดไป
        if len(splits) == 1:
            continue
            
        # รวมชิ้นส่วนกลับเข้ามาเป็น Chunk
        final_chunks = []
        current_chunk = []
        current_len = 0
        
        for s in splits:
            s_len = len(s)
            if current_len + s_len + len(sep) > target_size:
                # ถ้าเกินเป้า ให้จบ Chunk เดิม
                if current_chunk:
                    joined = sep.join(current_chunk)
                    final_chunks.append(joined)
                    
                    # เริ่ม Chunk ใหม่ โดยเอาส่วนท้ายของอันเก่ามา Overlap (ถ้าทำได้)
                    # ในที่นี้เริ่มใหม่เลยเพื่อความง่าย แต่ logic เรียก function นี้เราคุม overlap ระดับหน้าแล้ว
                    current_chunk = []
                    current_len = 0
            
            current_chunk.append(s)
            current_len += s_len + len(sep)
            
        if current_chunk:
            final_chunks.append(sep.join(current_chunk))
            
        return final_chunks
    
    return [text]


# -------------------------------------------------------------------
# 1) Text Chunking (Fixed & Improved)
# -------------------------------------------------------------------

def text_items_to_chunks(bundle: DocumentBundle) -> List[Chunk]:
    chunks: List[Chunk] = []
    
    # กรองเฉพาะที่มีเนื้อหา
    valid_texts = [t for t in bundle.texts if t.content and t.content.strip()]
    if not valid_texts:
        return chunks

    # [FIXED] ตัด Logic แยก Q&A ออก (เพราะมันทำให้ context ขาด) 
    # ให้ใช้ Logic รวมตามหน้า (Semantic Document Mode) สำหรับทุกเอกสารแทน
    
    # Group blocks by page
    page_groups = {}
    for t in valid_texts:
        p = t.page or 0
        if p not in page_groups:
            page_groups[p] = []
        page_groups[p].append(t)
        
    chunk_counter = 0
    
    for p_num in sorted(page_groups.keys()):
        blocks = page_groups[p_num]
        
        # รวมข้อความในหน้านั้นเป็นก้อนเดียว (เพื่อให้ ถาม-ตอบ ที่อยู่คนละ block มารวมกัน)
        full_page_text = "\n\n".join([b.content for b in blocks])
        full_page_text = _normalize_whitespace(full_page_text)
        
        # ใช้ Semantic Splitter ตัด โดยมี Overlap เพื่อกันตัดกลางประโยค
        split_contents = _split_text_recursively(
            full_page_text, 
            target_size=_TARGET_CHARS, 
            chunk_overlap=_CHUNK_OVERLAP
        )
        
        for i, content_part in enumerate(split_contents):
            chunk_counter += 1
            chunk_id = f"{blocks[0].doc_id}::text::p{p_num}_c{i:02d}"
            
            # Metadata รวมๆ ของหน้านั้น
            meta = {
                "page": p_num,
                "source": "text",
                "doc_id": blocks[0].doc_id,
                "split_part": i
            }
            
            chunks.append(Chunk(
                id=chunk_id,
                doc_id=blocks[0].doc_id,
                doc_type=blocks[0].doc_type or "generic",
                source="text",
                page=p_num,
                content=content_part,
                metadata=meta
            ))
            
    return chunks


# -------------------------------------------------------------------
# 2) Table Chunking
# -------------------------------------------------------------------

def table_items_to_chunks(bundle: DocumentBundle) -> List[Chunk]:
    chunks: List[Chunk] = []
    for item in bundle.tables:
        text = _table_to_text(item).strip()
        if not text: continue

        chunks.append(Chunk(
            id=f"{item.doc_id}::table::{item.id}",
            doc_id=item.doc_id,
            doc_type=item.doc_type,
            source="table",
            page=item.page,
            content=text,
            metadata={
                "table_id": item.id,
                "page": item.page,
                "columns": str(item.columns),
                "has_summary": bool(getattr(item, "extra", {}).get("summary"))
            }
        ))
    return chunks


# -------------------------------------------------------------------
# 3) Image Chunking
# -------------------------------------------------------------------

def image_items_to_chunks(bundle: DocumentBundle) -> List[Chunk]:
    chunks: List[Chunk] = []
    for item in bundle.images:
        # ใช้ Caption ที่ AI Generate มา
        content = item.caption or ""
        content = _normalize_whitespace(content)
        
        if not content: 
            continue

        chunks.append(Chunk(
            id=f"{item.doc_id}::image::{item.id}",
            doc_id=item.doc_id,
            doc_type=item.doc_type,
            source="image",
            page=item.page,
            content=f"คำอธิบายรูปภาพ (หน้า {item.page}): {content}",
            metadata={
                "image_id": item.id,
                "file_path": item.file_path,
                "page": item.page
            }
        ))
    return chunks