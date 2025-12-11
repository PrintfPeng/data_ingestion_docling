from __future__ import annotations

"""
pdf_parser.py

หน้าที่:
- เปิดไฟล์ PDF
- ดึงข้อความ (text) ออกจากทุกหน้า
- จัดลำดับการอ่าน (Reading Order) ให้ถูกต้อง
- เก็บพิกัด (bbox) ของแต่ละ block
- สร้าง DocumentMetadata + TextBlock ตาม schema
"""

from datetime import datetime
from pathlib import Path
from typing import List, Optional

import logging
import re
import fitz  # PyMuPDF

from .schema import (
    DocumentMetadata,
    TextBlock,
    IngestedDocument,
    BBox,
)

logger = logging.getLogger(__name__)


# -------------------------------------------------------------------
# Helper: doc_id
# -------------------------------------------------------------------
def _generate_doc_id(file_path: Path) -> str:
    return file_path.stem


# -------------------------------------------------------------------
# Helper: clean text / filter noise
# -------------------------------------------------------------------
_WORD_CHARS_PATTERN = re.compile(r"[A-Za-z0-9\u0E00-\u0E7F]")


def _clean_text(text: str) -> str:
    if not text:
        return ""
    # ลบ control characters (ยกเว้น newline)
    text = "".join(ch for ch in text if ch == "\n" or ch.isprintable())
    # แทนหลาย space/tab ด้วย space เดียว
    text = re.sub(r"[ \t]+", " ", text)
    # ลบ space ก่อน/หลัง newline
    text = re.sub(r" *\n *", "\n", text)
    return text.strip()


def _is_meaningful_text(text: str) -> bool:
    if not text:
        return False
    # นับตัวอักษรที่มีความหมาย
    matches = _WORD_CHARS_PATTERN.findall(text)
    if len(matches) < 2:
        return False
    return True


# -------------------------------------------------------------------
# Extract text blocks จากหนึ่งหน้า
# -------------------------------------------------------------------
def _extract_text_blocks_from_page(
    pdf_page: fitz.Page,
    doc_id: str,
    page_number: int,
    start_index: int = 0,
) -> List[TextBlock]:
    """
    ดึง text blocks โดยเรียงลำดับ Reading Order (บนลงล่าง, ซ้ายไปขวา)
    """
    try:
        # get_text("dict") คืนค่า blocks พร้อมพิกัดและรายละเอียด
        page_dict = pdf_page.get_text("dict", sort=True) 
        # sort=True ใน PyMuPDF ช่วยจัดเรียงเบื้องต้น แต่เราจะ sort เองอีกทีเพื่อความชัวร์
    except Exception as e:
        logger.warning(
            "[pdf_parser] get_text('dict') failed on page %d: %r, fallback to plain text",
            page_number,
            e,
        )
        txt = pdf_page.get_text("text") or ""
        txt = _clean_text(txt)
        if not _is_meaningful_text(txt):
            return []

        block_id = f"txt_{start_index + 1:04d}"
        bbox: BBox = (0.0, 0.0, 0.0, 0.0)
        return [
            TextBlock(
                id=block_id,
                doc_id=doc_id,
                page=page_number,
                content=txt,
                section=None,
                category=None,
                bbox=bbox,
                extra={"avg_font_size": None, "is_heading": False},
            )
        ]

    raw_blocks = page_dict.get("blocks", []) or []

    # Filter เฉพาะ Text Block (type=0) และตัด Image Block (type=1) ออก
    text_blocks_raw = [b for b in raw_blocks if b.get("type") == 0]

    # --- Logic จัดเรียง Reading Order (สำคัญมากสำหรับ RAG) ---
    # เรียงตามแกน Y (บนลงล่าง) ก่อน แล้วค่อยแกน X (ซ้ายไปขวา)
    # ใช้ tolerance เล็กน้อย (เช่น 3-5 pixel) เพื่อจัดกลุ่มบรรทัดเดียวกัน
    text_blocks_raw.sort(key=lambda b: (round(b["bbox"][1]), b["bbox"][0]))

    # คำนวณ Avg Font Size ของทั้งหน้า
    all_font_sizes: List[float] = []
    for block in text_blocks_raw:
        for line in block.get("lines", []) or []:
            for span in line.get("spans", []) or []:
                size = span.get("size")
                if size:
                    all_font_sizes.append(float(size))
    
    page_avg_font_size: Optional[float] = None
    if all_font_sizes:
        page_avg_font_size = sum(all_font_sizes) / len(all_font_sizes)

    text_blocks: List[TextBlock] = []
    current_index = start_index

    for block in text_blocks_raw:
        x0, y0, x1, y1 = block.get("bbox", (0.0, 0.0, 0.0, 0.0))
        
        lines = block.get("lines", []) or []
        spans_text: List[str] = []
        font_sizes: List[float] = []

        for line in lines:
            for span in line.get("spans", []) or []:
                text = span.get("text", "")
                if text and text.strip():
                    spans_text.append(text)
                    size = span.get("size")
                    if size:
                        font_sizes.append(float(size))

        if not spans_text:
            continue

        # รวม text ใน block
        content_raw = " ".join(spans_text)
        content = _clean_text(content_raw)
        
        if not _is_meaningful_text(content):
            continue

        avg_font_size: Optional[float] = None
        if font_sizes:
            avg_font_size = sum(font_sizes) / len(font_sizes)

        # Heuristic: Heading Detection
        is_heading = False
        if avg_font_size and page_avg_font_size:
            if avg_font_size >= page_avg_font_size * 1.25 and len(content) < 150:
                is_heading = True

        current_index += 1
        block_id = f"txt_{current_index:04d}"
        
        # ปัดทศนิยม BBox เพื่อความสวยงามและลดขนาด JSON
        bbox: BBox = (
            round(float(x0), 2), 
            round(float(y0), 2), 
            round(float(x1), 2), 
            round(float(y1), 2)
        )

        text_block = TextBlock(
            id=block_id,
            doc_id=doc_id,
            page=page_number,
            content=content,
            section=None,
            category=None,
            bbox=bbox,
            extra={
                "avg_font_size": round(avg_font_size, 2) if avg_font_size else None,
                "is_heading": is_heading,
            },
        )
        text_blocks.append(text_block)

    return text_blocks


# -------------------------------------------------------------------
# main parse function
# -------------------------------------------------------------------
def parse_pdf(
    file_path: str | Path,
    doc_type: str = "generic",
    doc_id: Optional[str] = None,
    source: str = "uploaded",
) -> IngestedDocument:
    path = Path(file_path)

    if not path.exists():
        raise FileNotFoundError(f"PDF file not found: {path}")

    logger.info("[pdf_parser] Parsing PDF: %s", path)

    pdf_doc = fitz.open(path)

    try:
        if doc_id is None:
            doc_id = _generate_doc_id(path)

        metadata = DocumentMetadata(
            doc_id=doc_id,
            file_name=path.name,
            doc_type=doc_type,
            page_count=pdf_doc.page_count,
            ingested_at=datetime.utcnow().isoformat(),
            source=source,
        )

        all_text_blocks: List[TextBlock] = []
        current_index = 0

        for page_index in range(pdf_doc.page_count):
            page = pdf_doc[page_index]
            page_number = page_index + 1
            
            page_text_blocks = _extract_text_blocks_from_page(
                pdf_page=page,
                doc_id=doc_id,
                page_number=page_number,
                start_index=current_index,
            )
            all_text_blocks.extend(page_text_blocks)
            current_index += len(page_text_blocks)

        logger.info(
            "[pdf_parser] Parsed doc_id=%s, pages=%d, text_blocks=%d",
            doc_id,
            pdf_doc.page_count,
            len(all_text_blocks),
        )

        return IngestedDocument(
            metadata=metadata,
            texts=all_text_blocks,
            tables=[],
            images=[],
        )

    finally:
        pdf_doc.close()

if __name__ == "__main__":
    import json
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("pdf_path")
    parser.add_argument("--doc-type", default="generic")
    args = parser.parse_args()

    doc = parse_pdf(args.pdf_path, doc_type=args.doc_type)
    print(json.dumps(doc.to_dict(), ensure_ascii=False, indent=2))