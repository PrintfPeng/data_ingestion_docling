from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional

import re
from pydantic import BaseModel, Field

from ..models import DocumentBundle, TableItem


class Chunk(BaseModel):
    """
    หนึ่งชิ้นข้อมูลที่เราจะส่งเข้า Vector DB
    - id        : ต้อง unique ทั่วทุก doc
    - doc_id    : ไว้ filter ว่าเป็นเอกสารใด
    - doc_type  : เช่น bank_statement, generic_doc ฯลฯ
    - source    : text / table / image
    - page      : หน้าเอกสารหลัก (ใช้ page แรกของ chunk ถ้าครอบคลุมหลายหน้า)
    - content   : เนื้อหาที่จะฝังเป็น embedding
    - metadata  : ข้อมูลเสริม เช่น block_ids, table_id, bbox ฯลฯ
    """

    id: str
    doc_id: str
    doc_type: str
    source: Literal["text", "table", "image"]
    page: Optional[int] = None
    content: str
    metadata: Dict[str, Any] = Field(default_factory=dict)


# ค่าเป้าหมายของขนาด chunk (สำหรับ text)
_TARGET_CHARS = 800      # อยากได้ประมาณนี้
_MAX_CHARS = 1200        # สูงสุดไม่เกินนี้
_MIN_CHARS = 200         # ถ้าต่ำกว่านี้จะพยายาม merge กับอันถัดไป


# -------------------------------------------------------------------
# Helper สำหรับ Table / Q&A detection
# -------------------------------------------------------------------

def _table_to_text(table: TableItem) -> str:
    """
    แปลง TableItem เป็น text แบบอ่านออกสำหรับ embedding
    - จำกัดจำนวนแถวไม่ให้ยาวเกินไป
    """
    header = f"Table {table.name} (page {table.page})"
    col_line = " | ".join(table.columns)

    row_lines: List[str] = []
    max_rows = min(len(table.rows), 10)  # กันไม่ให้ตารางใหญ่เกิน
    for row in table.rows[:max_rows]:
        # เผื่อบาง cell เป็น None
        safe_row = [str(c) if c is not None else "" for c in row]
        row_lines.append(" | ".join(safe_row))

    body = "\n".join(row_lines)
    text = f"{header}\nColumns: {col_line}\nRows:\n{body}"
    return text


def _is_qna_bundle(bundle: DocumentBundle) -> bool:
    """
    ตรวจคร่าว ๆ ว่าเอกสารนี้มี pattern ถาม: / ตอบ: เยอะไหม
    ถ้าใช่ → เราจะไม่ merge block เพื่อรักษา granularity ของ Q&A
    """
    count = 0
    for t in bundle.texts:
        if not t.content:
            continue
        c = t.content.replace(" ", "")
        if "ถาม:" in c or "ตอบ:" in c:
            count += 1
            if count >= 2:
                # มีอย่างน้อย 2 block ที่ดูเป็น Q&A → พอจะเชื่อว่าเป็นข้อสอบ/โจทย์
                return True
    return False


def _normalize_whitespace(text: str) -> str:
    """
    ล้าง whitespace แปลก ๆ ให้เหลือประมาณหนึ่ง:
    - แทนหลายช่องว่างด้วย space เดียว
    - ตัด space หน้า/หลัง
    """
    if not text:
        return ""
    # ไม่ไปยุ่งกับ newline มากนัก แค่ normalize space
    text = re.sub(r"[ \t]+", " ", text)
    return text.strip()


# -------------------------------------------------------------------
# 1) text → chunks (ฉลาดขึ้น)
# -------------------------------------------------------------------

def text_items_to_chunks(bundle: DocumentBundle) -> List[Chunk]:
    """
    แปลง text items ใน DocumentBundle เป็น chunks สำหรับ vector DB

    พฤติกรรม:
    - ถ้าเอกสารดูเหมือน Q&A (มี 'ถาม:' / 'ตอบ:' หลายบล็อค)
      → ไม่ merge แต่ละ text block = 1 chunk (ละเอียดไว้ก่อน)
    - ถ้าเป็นเอกสารทั่วไป
      → merge text blocks ที่ต่อเนื่องกัน เป็น chunk ขนาด ~800 ตัวอักษร
    """
    chunks: List[Chunk] = []
    texts = [t for t in bundle.texts if t.content]

    if not texts:
        return chunks

    is_qna = _is_qna_bundle(bundle)

    if is_qna:
        # --------- Q&A mode: ไม่ merge เพื่อให้แม็ปถาม/ตอบได้ชัด ---------
        for item in texts:
            content = _normalize_whitespace(item.content)
            if not content:
                continue

            chunk = Chunk(
                id=f"{item.doc_id}::text::{item.id}",
                doc_id=item.doc_id,
                doc_type=item.doc_type,
                source="text",
                page=item.page,
                content=content,
                metadata={
                    "block_id": item.id,
                    "section": item.section,
                    "bbox": item.bbox,
                    "page": item.page,
                    "doc_type": item.doc_type,
                },
            )
            chunks.append(chunk)

        return chunks

    # --------- Document mode: merge block เป็น chunk ขนาดเหมาะ ---------

    # สมมติว่า bundle.texts ถูกจัดเรียงตามลำดับการอ่านอยู่แล้ว
    buffer_contents: List[str] = []
    buffer_items: List[Any] = []
    current_len = 0
    chunk_index = 0

    def flush_buffer() -> None:
        nonlocal buffer_contents, buffer_items, current_len, chunk_index, chunks

        if not buffer_items:
            return

        merged_content = "\n".join(buffer_contents).strip()
        if not merged_content:
            buffer_contents = []
            buffer_items = []
            current_len = 0
            return

        first = buffer_items[0]
        last = buffer_items[-1]

        chunk_index += 1
        chunk_id = f"{first.doc_id}::text::chunk_{chunk_index:04d}"

        pages = [it.page for it in buffer_items if it.page is not None]
        page_main = pages[0] if pages else None

        metadata = {
            # เก็บ block ids ทั้งหมด (จะถูกแปลงเป็น string ตอน index)
            "block_ids": [it.id for it in buffer_items],
            "section_start": buffer_items[0].section,
            "section_end": buffer_items[-1].section,
            "page": page_main,
            "page_start": pages[0] if pages else None,
            "page_end": pages[-1] if pages else None,
            "doc_type": first.doc_type,
        }

        chunk = Chunk(
            id=chunk_id,
            doc_id=first.doc_id,
            doc_type=first.doc_type,
            source="text",
            page=page_main,
            content=merged_content,
            metadata=metadata,
        )
        chunks.append(chunk)

        # reset buffer
        buffer_contents = []
        buffer_items = []
        current_len = 0

    for item in texts:
        content = _normalize_whitespace(item.content)
        if not content:
            continue

        c_len = len(content)

        # ถ้า buffer ว่าง เริ่มใหม่เลย
        if not buffer_items:
            buffer_items = [item]
            buffer_contents = [content]
            current_len = c_len
            continue

        # ถ้าใส่อีกแล้วเกิน MAX_CHARS → flush buffer ก่อน
        if current_len + c_len > _MAX_CHARS:
            flush_buffer()
            # เริ่ม chunk ใหม่ด้วย item นี้
            buffer_items = [item]
            buffer_contents = [content]
            current_len = c_len
        else:
            # ยังไม่เกิน → ใส่ต่อใน chunk เดียวกัน
            buffer_items.append(item)
            buffer_contents.append(content)
            current_len += c_len

        # ถ้าถึงระดับ target แล้ว และยังมี buffer อยู่
        # ปล่อย flush เพื่อไม่ให้ chunk โตเกินเหตุ
        if current_len >= _TARGET_CHARS:
            flush_buffer()

    # เหลืออะไรใน buffer ก็ flush ท้ายสุด
    flush_buffer()

    return chunks


# -------------------------------------------------------------------
# 2) table → chunks
# -------------------------------------------------------------------

def table_items_to_chunks(bundle: DocumentBundle) -> List[Chunk]:
    """
    แปลง TableItem ใน DocumentBundle เป็น chunks
    - ไม่ merge ข้ามตาราง (แต่ละตาราง = 1 chunk)
    """
    chunks: List[Chunk] = []

    for item in bundle.tables:
        text_representation = _table_to_text(item).strip()
        if not text_representation:
            continue

        chunk = Chunk(
            id=f"{item.doc_id}::table::{item.id}",
            doc_id=item.doc_id,
            doc_type=item.doc_type,
            source="table",
            page=item.page,
            content=text_representation,
            metadata={
                "table_id": item.id,
                "name": item.name,
                "columns": item.columns,
                "bbox": item.bbox,
                "page": item.page,
                "doc_type": item.doc_type,
            },
        )
        chunks.append(chunk)

    return chunks


# -------------------------------------------------------------------
# 3) image → chunks
# -------------------------------------------------------------------

def image_items_to_chunks(bundle: DocumentBundle) -> List[Chunk]:
    """
    แปลง image items เป็น chunks
    - ใช้ caption เป็น content (ถ้ามี)
    """
    chunks: List[Chunk] = []

    for item in bundle.images:
        if not item.caption:
            continue

        caption = _normalize_whitespace(item.caption)
        if not caption:
            continue

        chunk = Chunk(
            id=f"{item.doc_id}::image::{item.id}",
            doc_id=item.doc_id,
            doc_type=item.doc_type,
            source="image",
            page=item.page,
            content=caption,
            metadata={
                "image_id": item.id,
                "file_path": item.file_path,
                "bbox": item.bbox,
                "page": item.page,
                "doc_type": item.doc_type,
            },
        )
        chunks.append(chunk)

    return chunks
