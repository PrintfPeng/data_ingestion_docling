from __future__ import annotations

"""
pdf_parser.py

หน้าที่:
- เปิดไฟล์ PDF
- ดึงข้อความ (text) ออกจากทุกหน้า
- เก็บพิกัด (bbox) ของแต่ละ block
- สร้าง DocumentMetadata + TextBlock ตาม schema
- คืนค่าเป็น IngestedDocument (ยังไม่มี table / image ในเฟสนี้)
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
    """
    สร้าง doc_id พื้นฐานจากชื่อไฟล์ (ไม่ต้องซับซ้อนมาก)
    เช่น sample.pdf -> sample
    """
    return file_path.stem


# -------------------------------------------------------------------
# Helper: clean text / filter noise
# -------------------------------------------------------------------
_WORD_CHARS_PATTERN = re.compile(r"[A-Za-z0-9\u0E00-\u0E7F]")


def _clean_text(text: str) -> str:
    """
    ทำความสะอาดข้อความเบื้องต้น:
    - แทน multiple spaces / tab เป็น space เดียว
    - ตัด space หน้า/หลัง
    - ลบ control characters แปลก ๆ
    """
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
    """
    กรอง block ที่เป็น noise:
    - ต้องมีตัวอักษร / ตัวเลข / ตัวไทยอย่างน้อย 2 ตัว
    - ถ้าสั้นมาก ๆ (< 3 ตัว) และไม่มีตัวสำคัญ → ตัดทิ้ง
    """
    if not text:
        return False

    # นับตัวอักษรที่มีความหมาย
    matches = _WORD_CHARS_PATTERN.findall(text)
    if len(matches) < 2:
        return False

    # ถ้ายาวน้อยกว่า 3 ตัวอักษรแต่มีตัวเลข/ไทย/อังกฤษสองตัว อาจจะเป็นรหัส/เลขหน้า → แล้วแต่จะเก็บ
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
    ดึง text blocks จากหน้าเดียวของ PDF
    ใช้ page.get_text("dict") เพื่อได้ทั้ง text + bbox + font size

    :param pdf_page: fitz.Page
    :param doc_id: ไอดีเอกสาร
    :param page_number: เลขหน้า (เริ่ม 1)
    :param start_index: index เริ่มต้นสำหรับ running id
    :return: list[TextBlock]
    """
    try:
        page_dict = pdf_page.get_text("dict")
    except Exception as e:  # noqa: BLE001
        logger.warning(
            "[pdf_parser] get_text('dict') failed on page %d: %r, fallback to plain text",
            page_number,
            e,
        )
        # fallback: ดึง text ทั้งหน้าแบบ plain แล้วใส่เป็น block เดียว
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

    blocks = page_dict.get("blocks", []) or []

    # 1) เก็บ font size ทั้งหน้าก่อน เพื่อคำนวณ avg / median ให้รู้ว่าบล็อคไหนใหญ่กว่าปกติ
    all_font_sizes: List[float] = []
    for block in blocks:
        for line in block.get("lines", []) or []:
            for span in line.get("spans", []) or []:
                size = span.get("size")
                if size:
                    try:
                        all_font_sizes.append(float(size))
                    except (TypeError, ValueError):
                        continue

    page_avg_font_size: Optional[float] = None
    if all_font_sizes:
        page_avg_font_size = sum(all_font_sizes) / len(all_font_sizes)

    text_blocks: List[TextBlock] = []
    current_index = start_index

    for block in blocks:
        # บาง block อาจไม่มี "lines" (เช่น รูปภาพ) ข้ามไป
        if "lines" not in block:
            continue

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
                        try:
                            font_sizes.append(float(size))
                        except (TypeError, ValueError):
                            continue

        # ถ้า block นี้ไม่มีข้อความที่มีเนื้อ ก็ข้ามไป
        if not spans_text:
            continue

        # รวม text ทั้ง block เป็นข้อความเดียว แล้วทำความสะอาด
        content_raw = " ".join(spans_text)
        content = _clean_text(content_raw)
        if not _is_meaningful_text(content):
            continue

        avg_font_size: Optional[float] = None
        if font_sizes:
            avg_font_size = sum(font_sizes) / len(font_sizes)

        # heuristic: ถ้า font ใหญ่กว่าค่าเฉลี่ยของหน้าเยอะ ๆ ให้ mark เป็น heading
        is_heading = False
        if avg_font_size and page_avg_font_size:
            if avg_font_size >= page_avg_font_size * 1.25 and len(content) < 120:
                is_heading = True

        current_index += 1
        block_id = f"txt_{current_index:04d}"
        bbox: BBox = (float(x0), float(y0), float(x1), float(y1))

        text_block = TextBlock(
            id=block_id,
            doc_id=doc_id,
            page=page_number,
            content=content,
            section=None,        # ยังไม่รู้ section (ให้ segmenter ทำต่อในเฟสหน้า)
            category=None,       # ยังไม่จัด category (ให้ categorizer ทำต่อ)
            bbox=bbox,
            extra={
                "avg_font_size": avg_font_size,
                "page_avg_font_size": page_avg_font_size,
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
    """
    ฟังก์ชันหลัก: แปลง PDF 1 ไฟล์ -> IngestedDocument (metadata + text blocks)

    - ยังไม่ดึงตาราง (ให้ table_extractor จัดการในเฟสถัดไป)
    - ยังไม่ดึงรูป (ให้ image_extractor จัดการในเฟสถัดไป)

    :param file_path: path ไปยัง PDF
    :param doc_type: ประเภทเอกสาร เช่น "bank_statement", "receipt", "invoice"
    :param doc_id: ถ้าไม่ระบุ จะสร้างจากชื่อไฟล์
    :param source: แหล่งที่มา เช่น "uploaded"
    :return: IngestedDocument
    """
    path = Path(file_path)

    if not path.exists():
        raise FileNotFoundError(f"PDF file not found: {path}")

    logger.info("[pdf_parser] Parsing PDF: %s", path)

    # เปิดเอกสารด้วย PyMuPDF
    pdf_doc = fitz.open(path)

    try:
        if doc_id is None:
            doc_id = _generate_doc_id(path)

        # สร้าง metadata
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

        # loop ทุกหน้า
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

        # คืนค่า document ที่มี metadata + text ทั้งหมด
        ingested = IngestedDocument(
            metadata=metadata,
            texts=all_text_blocks,
            tables=[],
            images=[],
        )
        return ingested

    finally:
        pdf_doc.close()


# เผื่ออยากรันทดสอบจาก command line โดยตรง
if __name__ == "__main__":
    import json
    import argparse

    parser = argparse.ArgumentParser(description="Parse PDF into structured text blocks.")
    parser.add_argument("pdf_path", help="Path to PDF file")
    parser.add_argument(
        "--doc-type",
        default="generic",
        help="Document type (e.g., bank_statement, receipt, invoice)",
    )
    args = parser.parse_args()

    doc = parse_pdf(args.pdf_path, doc_type=args.doc_type)
    # สมมติ IngestedDocument มีเมธอด .to_dict() ตาม schema เดิม
    print(json.dumps(doc.to_dict(), ensure_ascii=False, indent=2))
