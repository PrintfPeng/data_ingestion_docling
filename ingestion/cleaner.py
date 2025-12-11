from __future__ import annotations

"""
cleaner.py

Data Cleaning Engine ขั้นพื้นฐานสำหรับ:
- TextBlock: ล้าง whitespace, ตัด block ว่าง/ขยะ, ติด metadata เพิ่ม
- TableBlock: strip ช่องว่าง, ลบคอลัมน์/แถวที่ว่างเปล่า, normalize โครงสร้าง

ไฟล์นี้เน้น:
- ทำความสะอาดแบบ "ไม่ทำลายข้อมูลสำคัญ"
- เก็บ info เดิมไว้ใน extra.cleaning เผื่อ debug ทีหลัง
"""

from typing import List, Dict, Any
import re

from .schema import TextBlock, TableBlock

# -------------------------------------------------------------------
# Regex / helper พื้นฐาน
# -------------------------------------------------------------------

# ลบ control chars ทั่วไป (เว้น \t, \n, \r)
CONTROL_CHAR_RE = re.compile(r"[\x00-\x08\x0B\x0C\x0E-\x1F]")
# zero width / no-break space ที่ชอบติดมาจาก PDF / OCR
ZERO_WIDTH_RE = re.compile(r"[\u200B-\u200D\uFEFF]")
NBSP_RE = re.compile(r"\u00A0")

# ยุบ whitespace ภายในบรรทัด (เว้น newline)
INLINE_WS_RE = re.compile(r"[ \t\r\f\v]+")
# คำที่ถือว่ามี “ตัวอักษรสำคัญ” (อังกฤษ, เลข, ไทย)
WORD_CHARS_RE = re.compile(r"[A-Za-z0-9\u0E00-\u0E7F]")


def _normalize_text(s: str) -> str:
    """
    ล้าง control char + zero-width + NBSP + ยุบ whitespace ซ้ำในบรรทัด
    แต่ 'รักษาโครงสร้างบรรทัด' (ไม่บีบ newline ทิ้ง)
    """
    if not s:
        return ""

    # ลบ control char / zero-width / NBSP
    s = CONTROL_CHAR_RE.sub("", s)
    s = ZERO_WIDTH_RE.sub("", s)
    s = NBSP_RE.sub(" ", s)

    # แทน whitespace ภายในบรรทัดด้วย space เดียว
    # (ไม่ยุ่งกับ newline)
    s = INLINE_WS_RE.sub(" ", s)

    # ลบ space รอบ newline
    s = re.sub(r" *\n *", "\n", s)

    # บีบ newline ซ้ำ ให้เหลือไม่เกิน 2 บรรทัดติดกัน
    s = re.sub(r"\n{3,}", "\n\n", s)

    return s.strip()


def _is_noise_text(s: str) -> bool:
    """
    heuristic ง่าย ๆ ตรวจว่าข้อความน่าจะเป็น 'ขยะ' ไหม เช่น:
    - มีตัวอักษรสำคัญน้อยมาก
    - มีแต่ตัวเลขสั้น ๆ / page number / punctuation ล้วน
    """
    if not s:
        return True

    # นับตัวอักษรสำคัญ
    important = WORD_CHARS_RE.findall(s)
    if len(important) <= 1:
        return True

    # ถ้ายาวไม่เกิน 3 ตัว และไม่มีตัวอักษรไทย/อังกฤษ (เช่น "1", "-3-")
    if len(s) <= 3 and not re.search(r"[A-Za-z\u0E00-\u0E7F]", s):
        return True

    # ดัก pattern page number แบบพวก "- 3 -" หรือ "Page 3"
    if re.fullmatch(r"-?\s*\d+\s*-?", s):
        return True

    return False


# -------------------------------------------------------------------
# Cleaning: TextBlock
# -------------------------------------------------------------------

def clean_text_blocks(blocks: List[TextBlock]) -> List[TextBlock]:
    """
    ทำความสะอาด TextBlock:
    - ลบ control chars / zero-width / NBSP
    - ยุบ whitespace ภายในบรรทัด (แต่ไม่ทุบ newline ทิ้ง)
    - ตัด block ที่ว่างหรือดูเป็น noise
    - บันทึกข้อมูลก่อน/หลังใน extra.cleaning
    """
    cleaned: List[TextBlock] = []

    for b in blocks:
        original = b.content or ""
        normalized = _normalize_text(original)

        # เช็กว่ามีเนื้อหาจริงไหม
        if not normalized or _is_noise_text(normalized):
            # ทิ้ง block นี้ไป (พวก page number / ขยะ)
            continue

        b.content = normalized

        extra = dict(b.extra or {})
        cleaning_meta: Dict[str, Any] = dict(extra.get("cleaning", {}))
        cleaning_meta.update(
            {
                "original_length": len(original),
                "cleaned_length": len(normalized),
                "removed_chars": max(len(original) - len(normalized), 0),
                "was_noise": False,
            }
        )
        extra["cleaning"] = cleaning_meta
        b.extra = extra

        cleaned.append(b)

    return cleaned


# -------------------------------------------------------------------
# Cleaning: TableBlock
# -------------------------------------------------------------------

def _clean_table_cell(cell: Any) -> str:
    """ทำความสะอาดข้อความใน cell ตาราง (รับได้ทั้ง str / None / อื่น ๆ)"""
    if cell is None:
        return ""
    return _normalize_text(str(cell))


def clean_table_blocks(tables: List[TableBlock]) -> List[TableBlock]:
    """
    ทำความสะอาด TableBlock:
    - strip / normalize whitespace ใน header + rows
    - padding header/rows ให้จำนวน column เท่ากัน
    - ลบคอลัมน์ที่ว่างทุก cell
    - ลบแถวที่ว่างทุก cell
    - เก็บ metadata การเปลี่ยนแปลงไว้ใน extra.cleaning
    """
    cleaned_tables: List[TableBlock] = []

    for tb in tables:
        original_header = list(getattr(tb, "header", []) or [])
        original_rows = list(getattr(tb, "rows", []) or [])

        # 1) clean text ใน header / rows
        header_clean = [_clean_table_cell(h) for h in original_header]
        rows_clean = [[_clean_table_cell(c) for c in (row or [])] for row in original_rows]

        if header_clean or rows_clean:
            # 2) หา col_count สูงสุด แล้ว pad ทุกแถว/หัวให้เท่ากัน
            col_count = 0
            if header_clean:
                col_count = max(col_count, len(header_clean))
            for r in rows_clean:
                col_count = max(col_count, len(r))

            header_padded = header_clean + [""] * (col_count - len(header_clean))
            rows_padded = [
                (r + [""] * (col_count - len(r))) for r in rows_clean
            ]

            # 3) ลบคอลัมน์ที่ว่างทุก cell
            keep_col_idx = []
            for idx in range(col_count):
                col_vals = [header_padded[idx]] + [r[idx] for r in rows_padded]
                if any(v.strip() for v in col_vals):
                    keep_col_idx.append(idx)

            header_final = [header_padded[i] for i in keep_col_idx]
            rows_final = [[row[i] for i in keep_col_idx] for row in rows_padded]
        else:
            header_final = header_clean
            rows_final = rows_clean

        # 4) ลบแถวว่าง
        rows_final = [r for r in rows_final if any(c.strip() for c in r)]

        tb.header = header_final
        tb.rows = rows_final

        extra = dict(tb.extra or {})
        cleaning_meta: Dict[str, Any] = dict(extra.get("cleaning", {}))
        cleaning_meta.update(
            {
                "original_row_count": len(original_rows),
                "cleaned_row_count": len(rows_final),
                "original_header_len": len(original_header),
                "cleaned_header_len": len(header_final),
            }
        )
        extra["cleaning"] = cleaning_meta
        tb.extra = extra

        cleaned_tables.append(tb)

    return cleaned_tables
