from __future__ import annotations

"""
cleaner.py
"""

from typing import List, Dict, Any, Optional
import re

from .schema import TextBlock, TableBlock

# --- Regex เดิม ---
CONTROL_CHAR_RE = re.compile(r"[\x00-\x08\x0B\x0C\x0E-\x1F]")
ZERO_WIDTH_RE = re.compile(r"[\u200B-\u200D\uFEFF]")
NBSP_RE = re.compile(r"\u00A0")
INLINE_WS_RE = re.compile(r"[ \t\r\f\v]+")
WORD_CHARS_RE = re.compile(r"[A-Za-z0-9\u0E00-\u0E7F]")


def _normalize_text(s: str) -> str:
    if not s:
        return ""
    s = CONTROL_CHAR_RE.sub("", s)
    s = ZERO_WIDTH_RE.sub("", s)
    s = NBSP_RE.sub(" ", s)
    s = INLINE_WS_RE.sub(" ", s)
    s = re.sub(r" *\n *", "\n", s)
    s = re.sub(r"\n{3,}", "\n\n", s)
    return s.strip()


def _is_noise_text(s: str) -> bool:
    if not s:
        return True
    important = WORD_CHARS_RE.findall(s)
    if len(important) <= 1:
        return True
    if len(s) <= 3 and not re.search(r"[A-Za-z\u0E00-\u0E7F]", s):
        return True
    if re.fullmatch(r"-?\s*\d+\s*-?", s): # page number pattern
        return True
    return False


# -------------------------------------------------------------------
# Cleaning: TextBlock
# -------------------------------------------------------------------
def clean_text_blocks(blocks: List[TextBlock]) -> List[TextBlock]:
    cleaned: List[TextBlock] = []
    for b in blocks:
        original = b.content or ""
        normalized = _normalize_text(original)

        if not normalized or _is_noise_text(normalized):
            continue

        b.content = normalized
        # Update metadata for debugging
        extra = dict(b.extra or {})
        extra["cleaning"] = {
            "original_len": len(original),
            "cleaned_len": len(normalized)
        }
        b.extra = extra
        cleaned.append(b)

    return cleaned


# -------------------------------------------------------------------
# Feature ใหม่: Overlap Removal (Text vs Table)
# -------------------------------------------------------------------
def remove_text_inside_tables(
    text_blocks: List[TextBlock], 
    table_blocks: List[TableBlock]
) -> List[TextBlock]:
    """
    ลบ TextBlock ที่อยู่ในพื้นที่ของ TableBlock เพื่อป้องกันข้อมูลซ้ำซ้อน
    """
    if not table_blocks:
        return text_blocks

    final_blocks = []
    
    # Group tables by page for efficiency
    tables_by_page = {}
    for tb in table_blocks:
        tables_by_page.setdefault(tb.page, []).append(tb)

    for txt in text_blocks:
        if not txt.bbox:
            final_blocks.append(txt)
            continue

        page_tables = tables_by_page.get(txt.page, [])
        is_overlap = False
        
        # Calculate text center point
        tx0, ty0, tx1, ty1 = txt.bbox
        cx, cy = (tx0 + tx1) / 2, (ty0 + ty1) / 2

        for tb in page_tables:
            if not tb.bbox:
                continue
            bx0, by0, bx1, by1 = tb.bbox
            
            # Simple containment check (Text center inside Table box)
            # เพิ่ม margin เล็กน้อยเผื่อขอบ
            if (bx0 <= cx <= bx1) and (by0 <= cy <= by1):
                is_overlap = True
                break
        
        if not is_overlap:
            final_blocks.append(txt)

    return final_blocks


# -------------------------------------------------------------------
# Cleaning: TableBlock
# -------------------------------------------------------------------
def _clean_table_cell(cell: Any) -> str:
    if cell is None:
        return ""
    return _normalize_text(str(cell))

def clean_table_blocks(tables: List[TableBlock]) -> List[TableBlock]:
    cleaned_tables: List[TableBlock] = []

    for tb in tables:
        original_header = list(getattr(tb, "header", []) or [])
        original_rows = list(getattr(tb, "rows", []) or [])

        # Clean text inside cells
        header_clean = [_clean_table_cell(h) for h in original_header]
        rows_clean = [[_clean_table_cell(c) for c in (row or [])] for row in original_rows]

        # Normalize columns length (Padding)
        col_count = len(header_clean) if header_clean else 0
        for r in rows_clean:
            col_count = max(col_count, len(r))

        header_padded = header_clean + [""] * (col_count - len(header_clean))
        rows_padded = [(r + [""] * (col_count - len(r))) for r in rows_clean]

        # Remove empty rows
        rows_final = [r for r in rows_padded if any(c.strip() for c in r)]

        tb.header = header_padded
        tb.rows = rows_final
        cleaned_tables.append(tb)

    return cleaned_tables