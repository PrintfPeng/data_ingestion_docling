from __future__ import annotations

"""
table_extractor.py

หน้าที่:
- ใช้ Camelot อ่านตารางจาก PDF
- แปลงตารางแต่ละตัวเป็น TableBlock ตาม schema
- คืนค่า list[TableBlock] เพื่อไปใส่ใน IngestedDocument.tables ภายหลัง
"""

from pathlib import Path
from typing import List, Optional, Any, Tuple

import camelot
import pandas as pd

from .schema import TableBlock, BBox

# -------------------------------
# Config / Heuristics
# -------------------------------

MIN_ROWS = 2          # น้อยกว่านี้มักไม่ใช่ตารางจริง ๆ
MIN_COLS = 2
MAX_HEADER_SCAN_ROWS = 3  # สแกนบนสุดกี่แถวเพื่อหา header ที่แท้จริง


def _has_meaningful_text(s: str) -> bool:
    """เช็กว่ามีตัวอักษร/ตัวเลขจริง ๆ ไหม (กัน cell ที่เป็นขยะ whitespace)"""
    if s is None:
        return False
    s = str(s).strip()
    if not s:
        return False
    # ถ้ามีตัวอักษรหรือเลขถือว่ามีเนื้อ
    return any(ch.isalnum() for ch in s)


# -------------------------------
# Category detection
# -------------------------------

def _guess_table_category(columns: list[str], first_row: list[Any]) -> str:
    """
    เดา category ของตารางจาก header + แถวแรก
    """
    header_lower = " ".join(columns).lower()
    first_row_lower = " ".join(str(x) for x in first_row).lower()

    text_for_detect = header_lower + " " + first_row_lower

    # transaction-like
    if any(k in text_for_detect for k in ["date", "วันที่", "วันเดือนปี"]) and any(
        k in text_for_detect for k in ["amount", "ยอด", "ยอดเงิน", "จำนวนเงิน", "เงิน", "debit", "credit", "คงเหลือ"]
    ):
        return "transaction_table"

    # item list
    if any(k in text_for_detect for k in ["item", "description", "รายละเอียด", "รายการ", "product", "สินค้า"]):
        return "item_list"

    return "generic_table"


# -------------------------------
# DataFrame → header / rows
# -------------------------------

def _find_header_row_index(df: pd.DataFrame) -> int:
    """
    พยายามหา row index ที่น่าจะเป็น header:
    - ดูไม่เกิน MAX_HEADER_SCAN_ROWS แรก
    - เลือกแถวที่มี cell เป็นตัวหนังสือมากที่สุด
    """
    best_idx = 0
    best_score = -1

    max_scan = min(MAX_HEADER_SCAN_ROWS, len(df))

    for i in range(max_scan):
        row = df.iloc[i]
        score = 0
        for v in row:
            if _has_meaningful_text(v):
                score += 1
        if score > best_score:
            best_score = score
            best_idx = i

    return best_idx


def _dataframe_to_columns_rows(df: pd.DataFrame) -> Tuple[list[str], list[list[Any]]]:
    """
    แปลง DataFrame ที่ Camelot คืนมาให้เป็น (columns, rows)
    พร้อม heuristic:
    - ลบแถว/คอลัมน์ว่างล้วน
    - หา header row จากแถวบนสุดไม่กี่แถว
    """
    if df.empty:
        return [], []

    # แปลงทุก cell เป็น string แบบ "soft" ก่อน
    df_str = df.astype(str)

    # ลบแถวที่ทุก cell ว่าง
    mask_non_empty_row = df_str.apply(lambda r: any(_has_meaningful_text(c) for c in r), axis=1)
    df_str = df_str[mask_non_empty_row]
    if df_str.empty:
        return [], []

    # ลบคอลัมน์ที่ทุก cell ว่าง
    mask_non_empty_col = df_str.apply(lambda c: any(_has_meaningful_text(v) for v in c))
    df_str = df_str.loc[:, mask_non_empty_col]
    if df_str.empty:
        return [], []

    # หา header row index แบบฉลาด
    header_idx = _find_header_row_index(df_str)

    header_series = df_str.iloc[header_idx]
    header = [str(h).strip() for h in header_series.tolist()]

    # ส่วนที่เหลือเป็น data rows
    data_part = df_str.iloc[header_idx + 1 :]
    rows: list[list[Any]] = [list(r) for _, r in data_part.iterrows()]

    # ตัดแถวที่ว่างล้วนอีกที
    rows = [
        [str(c).strip() for c in row]
        for row in rows
        if any(_has_meaningful_text(c) for c in row)
    ]

    return header, rows


# -------------------------------
# Main extraction
# -------------------------------

def extract_tables(
    file_path: str | Path,
    doc_id: str,
    doc_type: str = "generic",
    pages: str = "all",
    flavor_priority: Optional[list[str]] = None,
) -> List[TableBlock]:
    """
    ดึงตารางจาก PDF 1 ไฟล์ทั้งหมด

    :param file_path: path ไปยัง PDF
    :param doc_id: ไอดีเอกสาร (ใช้เชื่อมกับ DocumentMetadata)
    :param doc_type: ประเภทเอกสาร (เผื่อใช้ logic เพิ่มเติมในอนาคต)
    :param pages: หน้า เช่น "1", "1,2,3", "1-3", "all"
    :param flavor_priority: ลำดับการลอง flavor ของ Camelot เช่น ["lattice", "stream"]
    :return: list[TableBlock]
    """
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"PDF file not found: {path}")

    # ค่า default: ลอง lattice ก่อน ถ้าไม่ได้ค่อยลอง stream
    if flavor_priority is None:
        flavor_priority = ["lattice", "stream"]

    all_tables: List[TableBlock] = []
    table_index = 0

    # ลองแต่ละ flavor ตามลำดับจนกว่าจะเจอตาราง "ใช้ได้"
    for flavor in flavor_priority:
        try:
            tables = camelot.read_pdf(
                str(path),
                pages=pages,
                flavor=flavor,
            )
        except Exception as e:
            # ถ้า flavor นี้ใช้ไม่ได้ (เช่น PDF ไม่มีเส้นตารางสำหรับ lattice) ก็ข้าม
            print(f"[table_extractor] Error using flavor='{flavor}': {e}")
            continue

        if tables.n == 0:
            # ไม่มีตารางใน flavor นี้
            continue

        for t in tables:
            df: pd.DataFrame = t.df

            columns, rows = _dataframe_to_columns_rows(df)

            # กรองตารางที่เล็กเกิน / ว่างเกิน
            if len(columns) < MIN_COLS or len(rows) < MIN_ROWS:
                continue

            first_row_for_category = rows[0] if rows else []
            category = _guess_table_category(columns, first_row_for_category)

            # พยายามดึง bbox ถ้ามี (บาง version ของ Camelot มี attribute _bbox)
            bbox: Optional[BBox] = None
            if hasattr(t, "_bbox") and t._bbox is not None:
                x1, y1, x2, y2 = t._bbox
                bbox = (float(x1), float(y1), float(x2), float(y2))

            table_index += 1
            table_id = f"tbl_{table_index:04d}"

            table_block = TableBlock(
                id=table_id,
                doc_id=doc_id,
                page=t.page,              # page index ที่ Camelot แยกได้ (เริ่ม 1)
                name=f"table_{table_index}",
                section=None,             # ภายหลังให้ segmenter / enricher ใส่
                category=category,
                columns=columns,
                rows=rows,
                bbox=bbox,
                extra={
                    "camelot_flavor": flavor,
                    "parsing_report": t.parsing_report,  # ใช้ debug ได้
                    "doc_type": doc_type,
                    "raw_shape": (df.shape[0], df.shape[1]),
                },
            )
            all_tables.append(table_block)

        # ถ้า flavor นี้เจอตาราง "ดีพอ" แล้ว ก็หยุดไม่ลอง flavor ถัดไป
        if all_tables:
            break

    return all_tables


# -------------------------------
# CLI test
# -------------------------------

if __name__ == "__main__":
    import json
    import argparse

    parser = argparse.ArgumentParser(
        description="Extract tables from PDF into TableBlock list."
    )
    parser.add_argument("pdf_path", help="Path to PDF file")
    parser.add_argument("--doc-id", help="Document ID (default: stem of file name)")
    parser.add_argument("--doc-type", default="generic", help="Document type")
    parser.add_argument("--pages", default="all", help="Pages to parse (e.g., '1', '1-3', 'all')")
    args = parser.parse_args()

    pdf_path = args.pdf_path
    doc_id = args.doc_id or Path(pdf_path).stem

    tables = extract_tables(
        file_path=pdf_path,
        doc_id=doc_id,
        doc_type=args.doc_type,
        pages=args.pages,
    )

    print(f"Extracted {len(tables)} tables.")
    data = [t.to_dict() for t in tables]
    print(json.dumps(data, ensure_ascii=False, indent=2))
