from __future__ import annotations

"""
table_extractor.py

หน้าที่:
- ใช้ Camelot อ่านตารางจาก PDF (วิธีเดิม)
- [NEW] ใช้ Custom Vision (Qwen-VL) อ่านตารางจากภาพ (วิธีใหม่สำหรับตารางซับซ้อน)
- [NEW] ทำความสะอาดภาษาไทย (ลบ \\n กลางคำ)
- [NEW] AI-Powered Classification: ใช้ LLM จำแนกหมวดหมู่ตารางแทน Regex
- แปลงตารางเป็น HTML (Display) และ Markdown (AI Context)
- คืนค่า list[TableBlock]
"""

import time
import io
import re
import hashlib # FIX TASK 2: Deterministic hashing
import base64
from pathlib import Path
from typing import List, Optional, Any, Tuple
from html.parser import HTMLParser

# [POLICY] ห้าม import pandas ที่ top-level
import camelot
import fitz  # PyMuPDF
from PIL import Image
# [CHANGE] ใช้ OpenAI Client
try:
    from openai import OpenAI
except ImportError:
    OpenAI = None

from .schema import TableBlock # FIX TASK 5: Removed unused BBox import
# [CHANGE] ลบการ import key เก่า
from dotenv import load_dotenv
import os

load_dotenv()

# -------------------------------
# Config / Heuristics
# -------------------------------

MIN_ROWS = 2
MIN_COLS = 2
MAX_HEADER_SCAN_ROWS = 3

# FIX TASK 6: Metadata Versioning
SCHEMA_VERSION = "tableblock_v1"
PARSER_VERSION = "2026-01"

# [CHANGE] เพิ่ม Timeout (แก้ตามสั่ง)
DEFAULT_TIMEOUT = 120.0

HEADER_PATTERNS = [
    r"ประวัติการศึกษา ",
    r"ประวัติการอบรม",
    r"ความสามารถทางภาษา",
]

# [CHANGE] Model Config
# โมเดลสำหรับอ่านภาพ (Vision)
VISION_MODEL = "qwen/qwen2.5-vl-32b-instruct"
# โมเดลสำหรับวิเคราะห์ข้อความ (Text)
TEXT_MODEL = os.getenv("CUSTOM_MODEL_NAME", "qwen/qwen-2.5-72b-instruct")

# -------------------------------
# Text Cleaning Helpers
# -------------------------------

def _clean_thai_text(text: Any) -> str:
    if text is None:
        return ""
    text = str(text).strip()
    
    # ลบ \n ที่อยู่ระหว่างตัวอักษรไทย
    text = re.sub(r'(?<=[\u0E00-\u0E7F])\s*[\r\n]+\s*(?=[\u0E00-\u0E7F])', '', text)
    text = re.sub(r'[\r\n]+', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'\.{3,}', '', text)
    
    return text.strip()


def _has_meaningful_text(s: str) -> bool:
    if s is None:
        return False
    s = str(s).strip()
    if not s:
        return False
    return any(ch.isalnum() for ch in s)


# -------------------------------
# [NEW] Content Hashing Helper (Row Content Only)
# -------------------------------
def _compute_row_content_hash(rows: list[list[str]]) -> str:
    """
    สร้าง Hash จากเนื้อหาใน Rows เท่านั้น (ไม่รวม Header/Columns)
    แก้ปัญหา Duplicate ที่ Header เขียนไม่เหมือนกัน หรือ Space ต่างกันเล็กน้อย
    """
    # Flatten rows, join text
    row_content = "".join(["".join(map(str, r)) for r in rows])
    # Normalize: ตัดช่องว่างทิ้งทั้งหมด เพื่อให้การเปรียบเทียบแม่นยำที่สุด
    row_content = re.sub(r"[\s\u200b]+", "", row_content).lower()
    
    return hashlib.md5(row_content.encode('utf-8')).hexdigest()


# -------------------------------
# [CRITICAL] HTML Parser for Vision Tables (Fixed Logic)
# -------------------------------

class SimpleTableParser(HTMLParser):
    """Parse HTML <table> เป็น columns + rows (ไม่พึ่ง Pandas)"""
    def __init__(self):
        super().__init__()
        self.in_table = False
        self.in_thead = False
        self.in_tbody = False
        self.in_tr = False
        self.in_th = False
        self.in_td = False
        self.current_text = []
        self.headers = []
        self.current_row = []
        self.rows = []
        # FIX TASK 1: Mark as legacy flag explicitly. Do not use for logic.
        self.has_complex_structure = False  
        self.has_complex_header = False     # Specific flag for header
        self.has_complex_body = False       # Specific flag for body
        
    def handle_starttag(self, tag, attrs):
        if tag == 'table':
            self.in_table = True
        elif tag == 'thead':
            self.in_thead = True
        elif tag == 'tbody':
            self.in_tbody = True
        elif tag == 'tr':
            self.in_tr = True
            self.current_row = []
        elif tag in ('th', 'td'):
            if tag == 'th':
                self.in_th = True
            else:
                self.in_td = True
            self.current_text = []
            
            # FIX TASK 1: Check rowspan/colspan accurately for Header vs Body
            attr_dict = {k.lower(): v for k, v in attrs}
            rowspan = attr_dict.get('rowspan', '1')
            colspan = attr_dict.get('colspan', '1')
            
            try:
                r_val = int(rowspan)
                c_val = int(colspan)
                
                # Check if we are physically or logically in the header
                # If we haven't found any rows yet, we treat this as header territory
                is_header_row = (not self.rows) 
                
                if r_val > 1 or c_val > 1:
                    # Legacy flag set for backward compat, but won't be used for decisions
                    self.has_complex_structure = True
                    
                    if is_header_row:
                        # Header merge: flag as complex header (if rowspan > 1)
                        # colspan in header is acceptable for flat extraction
                        if r_val > 1: 
                            self.has_complex_header = True
                    else:
                        # Body merge: flag as complex body (CRITICAL: Data extraction impossible)
                        self.has_complex_body = True
                        
            except ValueError:
                pass 
            
    def handle_endtag(self, tag):
        if tag == 'table':
            self.in_table = False
        elif tag == 'thead':
            self.in_thead = False
        elif tag == 'tbody':
            self.in_tbody = False
        elif tag == 'tr':
            self.in_tr = False
            # FIX TASK 2: Robust Header Detection
            # If we don't have headers yet, the FIRST row we see is the header.
            # Regardless of whether it's in <thead> or <tbody> (Vision OCR is messy).
            if not self.headers:
                self.headers = self.current_row
            else:
                if self.current_row: # Only add if not empty
                    self.rows.append(self.current_row)
            self.current_row = []
        elif tag in ('th', 'td'):
            self.in_th = False
            self.in_td = False
            text = ''.join(self.current_text).strip()
            self.current_row.append(text)
            self.current_text = []
            
    def handle_data(self, data):
        if self.in_th or self.in_td:
            self.current_text.append(data)
    
    def get_table_data(self) -> Tuple[list[str], list[list[str]]]:
        """
        Returns (columns, rows)
        """
        columns = self.headers
        data_rows = self.rows
        
        # Fallback if no rows extracted (but headers exist)
        if not columns and data_rows:
            columns = data_rows[0]
            data_rows = data_rows[1:]
        
        if not columns:
            return [], []
        
        # Normalize rows to match header length
        expected_len = len(columns)
        normalized_rows = []
        for row in data_rows:
            if len(row) > expected_len:
                normalized_rows.append(row[:expected_len])
            elif len(row) < expected_len:
                normalized_rows.append(row + [""] * (expected_len - len(row)))
            else:
                normalized_rows.append(row)
        
        return columns, normalized_rows


def parse_html_table(html: str) -> Tuple[list[str], list[list[str]], bool, bool]:
    """
    Parse HTML <table> → (columns, rows, has_complex_body, has_complex_header)
    FIX TASK 1: Return granular complexity flags
    """
    parser = SimpleTableParser()
    try:
        parser.feed(html)
        
        # FIX TASK 1: Return flags from parser state
        columns, rows = parser.get_table_data()
        
        # FIX TASK 3: Normalize invalid state (Silent failure prevention)
        # If body is complex, structured data is unreliable -> Force empty
        if parser.has_complex_body:
             return [], [], True, parser.has_complex_header
        
        # If columns exist but rows are empty, treat as lossy structure
        # (Prevent downstream from thinking it has a valid empty table)
        if columns and not rows:
             return [], [], True, parser.has_complex_header

        # Clean text
        columns = [_clean_thai_text(c) for c in columns]
        rows = [[_clean_thai_text(cell) for cell in row] for row in rows]
        
        return columns, rows, parser.has_complex_body, parser.has_complex_header
    except Exception as e:
        print(f"[table_extractor] HTML parse failed: {e}")
        # Parse error -> treat as complex body
        return [], [], True, False 


# -------------------------------
# Markdown Generation
# -------------------------------

def table_to_markdown(columns: list[str], rows: list[list[Any]]) -> str:
    """สร้าง Markdown มาตรฐานจาก columns + rows"""
    if not columns: 
        return ""
    
    lines = []
    lines.append("| " + " | ".join(str(c) for c in columns) + " |")
    lines.append("| " + " | ".join("---" for _ in columns) + " |")
    for row in rows:
        padded_row = list(row) + [""] * (len(columns) - len(row))
        lines.append("| " + " | ".join(str(c) for c in padded_row[:len(columns)]) + " |")
    
    return "\n".join(lines)


# -------------------------------
# HTML Generation (No Pandas)
# -------------------------------

def table_to_html(columns: list[str], rows: list[list[Any]]) -> str:
    if not columns: return ""
    html_parts = []
    html_parts.append('<table class="min-w-full text-sm text-left text-slate-600 border-collapse border border-slate-200">')
    html_parts.append('<thead class="bg-slate-100 text-slate-700 font-semibold"><tr>')
    for col in columns:
        html_parts.append(f'<th class="px-4 py-2 border border-slate-200">{col}</th>')
    html_parts.append('</tr></thead><tbody>')
    for row in rows:
        html_parts.append('<tr>')
        padded_row = list(row) + [""] * (len(columns) - len(row))
        for cell in padded_row[:len(columns)]:
            html_parts.append(f'<td class="px-4 py-2 border border-slate-200 align-top">{cell}</td>')
        html_parts.append('</tr>')
    html_parts.append('</tbody></table>')
    return ''.join(html_parts)


# -------------------------------
# Helpers
# -------------------------------

def _split_rows_by_header(rows: list[list[Any]]) -> list[tuple[str, list[list[Any]]]]:
    blocks = []
    current_header = "Generic Section"
    current_rows = []
    for r in rows:
        row_text = " ".join(str(c) for c in r)
        found_header = None
        for p in HEADER_PATTERNS:
            if re.search(p, row_text, re.IGNORECASE):
                found_header = row_text
                break
        if found_header:
            if current_rows:
                blocks.append((current_header, current_rows))
            current_header = _clean_thai_text(found_header)
            current_rows = []
        else:
            cleaned_row = [_clean_thai_text(c) for c in r]
            if any(_has_meaningful_text(c) for c in cleaned_row):
                current_rows.append(cleaned_row)
    if current_rows:
        blocks.append((current_header, current_rows))
    return blocks


# [CHANGE] Get Custom API Client with Timeout
def _get_llm_client() -> Optional[OpenAI]:
    api_key = os.getenv("CUSTOM_API_KEY")
    base_url = os.getenv("CUSTOM_API_BASE")
    if not api_key: return None
    try:
        # [CHANGE] เพิ่ม timeout ตามคำสั่ง
        return OpenAI(
            api_key=api_key, 
            base_url=base_url,
            timeout=DEFAULT_TIMEOUT
        )
    except Exception as e:
        print(f"[table_extractor] Init LLM Client Error: {e}")
        return None


def _truncate_html_safely(html: str, max_length: int = 4000) -> str:
    """
    FIX TASK 5: Harden HTML Truncation by Checking Context & Unclosed Tags
    """
    if len(html) <= max_length: return html
    
    last_tr_end = html.rfind('</tr>', 0, max_length)
    
    if last_tr_end != -1:
        truncated = html[:last_tr_end + 5]
    else:
        # Fallback: Truncate but avoid breaking tags
        truncated = html[:max_length]
        # FIX TASK 5: Prevent cutting mid-tag
        last_open_tag = truncated.rfind('<')
        last_close_tag = truncated.rfind('>')
        if last_open_tag > last_close_tag: # We are inside a tag
            truncated = truncated[:last_open_tag]

    # FIX TASK 5: Ensure TR is closed if opened
    if truncated.count('<tr>') > truncated.count('</tr>'):
        truncated += '</tr>'

    # Check context indices
    last_thead_open = truncated.rfind('<thead')
    last_thead_close = truncated.rfind('</thead>')
    
    last_tbody_open = truncated.rfind('<tbody')
    last_tbody_close = truncated.rfind('</tbody>')
    
    last_table_open = truncated.rfind('<table')
    last_table_close = truncated.rfind('</table>')
    
    # Close open tags in correct order (inner to outer)
    if last_thead_open != -1 and (last_thead_close == -1 or last_thead_close < last_thead_open):
        truncated += '</thead>'
    
    if last_tbody_open != -1 and (last_tbody_close == -1 or last_tbody_close < last_tbody_open):
        truncated += '</tbody>'
        
    if last_table_open != -1 and (last_table_close == -1 or last_table_close < last_table_open):
        truncated += '</table>'
    
    return truncated


def _summarize_table(client: OpenAI, markdown_table: str, is_html: bool = False) -> str:
    if not client: return ""
    truncated = _truncate_html_safely(markdown_table) if is_html else markdown_table[:4000]
    prompt = (
        f"{'หมายเหตุ: ตาราง HTML ซับซ้อน\n' if is_html else ''}"
        "สรุปใจความสำคัญของตารางนี้สั้นๆ (ไม่เกิน 3 บรรทัด):\n"
        f"ข้อมูล:\n{truncated}"
    )
    try:
        response = client.chat.completions.create(
            model=TEXT_MODEL,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=150,
            temperature=0.1,
            timeout=60.0
        )
        summary = response.choices[0].message.content.strip()
        
        # [🔥 แก้ตรงนี้] เพิ่ม Sleep 15 วินาที กัน Error 429
        print("   💤 Summarized. Cooling down API for 15s...")
        time.sleep(15) 
        
        return summary
    except Exception as e:
        print(f"[table_extractor] Summarization failed: {e}")
        return ""


def _extract_text_from_html_headers(html: str) -> str:
    if not html: return ""
    text_parts = []
    # FIX TASK 6: Ensure caption/th is extracted even if malformed
    caption = re.search(r'<caption[^>]*>(.*?)</caption>', html, re.IGNORECASE | re.DOTALL)
    if caption:
        text_parts.append(re.sub(r'<[^>]+>', '', caption.group(1)).strip())
    
    headers = re.findall(r'<th[^>]*>(.*?)</th>', html, re.IGNORECASE | re.DOTALL)
    for h in headers:
        clean_text = re.sub(r'<[^>]+>', ' ', h).strip()
        if clean_text:
            text_parts.append(clean_text)
            
    return " ".join(text_parts)


def _classify_category_with_llm(client: OpenAI, text_sample: str) -> str:
    """
    FIX TASK 6: Robust Classification Input
    """
    input_text = f"table_context: {text_sample}".strip()
    # FIX TASK 6: Ensure input is not empty/blind
    if len(input_text) < 50:
        input_text += " generic_table_hint"
        
    if not client: return "generic_table"
    safe_sample = input_text[:1000]
    
    prompt = (
        "คุณคือระบบจำแนกตาราง ระบุหมวดหมู่จาก: "
        "slogan_holder, parade, fancy, student_council, equipment, budget, schedule, staff, generic_table\n"
        f"ข้อมูล: '{safe_sample}'\n"
        "ตอบเฉพาะชื่อหมวดหมู่ภาษาอังกฤษ (snake_case) เท่านั้น:"
    )
    try:
        # [CHANGE] ใส่ try-except และ timeout
        response = client.chat.completions.create(
            model=TEXT_MODEL,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=50,
            temperature=0.0,
            timeout=30.0
        )
        category = response.choices[0].message.content.strip().lower()
        category = re.sub(r"[^a-z_]", "", category)
        return category if category else "generic_table"
    except Exception as e:
        print(f"[table_extractor] Classification failed: {e}")
        return "generic_table"


def _pil_image_to_base64(image: Image.Image) -> str:
    buffered = io.BytesIO()
    image.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode("utf-8")


def _extract_table_with_vision(client: OpenAI, image: Image.Image) -> str:
    prompt = "Extract table to HTML. Use only <table>, <thead>, <tbody>, <tr>, <th>, <td> tags. No markdown."
    try:
        b64_image = _pil_image_to_base64(image)
        
        response = client.chat.completions.create(
            model=VISION_MODEL,
            messages=[
                {
                    "role": "user", 
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/png;base64,{b64_image}"}
                        }
                    ]
                }
            ],
            max_tokens=2000,
            timeout=DEFAULT_TIMEOUT
        )
        html = response.choices[0].message.content.replace("```html", "").replace("```", "").strip()
        
        # [🔥 แก้ตรงนี้] เพิ่ม Sleep 15 วินาที หลังงาน Vision
        print("   💤 Vision extracted. Cooling down API for 15s...")
        time.sleep(15)

        return html if "<table" in html else ""
    except Exception as e:
        print(f"[table_extractor] Vision extraction failed: {e}")
        return ""


# -------------------------------
# DataFrame Helpers (Pandas On-Demand)
# -------------------------------

def _find_header_row_index(df) -> int:
    best_idx = 0
    best_score = -1
    max_scan = min(MAX_HEADER_SCAN_ROWS, len(df))
    for i in range(max_scan):
        row = df.iloc[i]
        score = sum(1 for v in row if _has_meaningful_text(v))
        if score > best_score:
            best_score = score
            best_idx = i
    return best_idx

def _dataframe_to_columns_rows(df) -> Tuple[list[str], list[list[Any]]]:
    import pandas as pd
    if df.empty: return [], []
    try: df = df.map(_clean_thai_text)
    except: df = df.applymap(_clean_thai_text)
    
    df_str = df.astype(str)
    mask_row = df_str.apply(lambda r: any(_has_meaningful_text(c) for c in r), axis=1)
    df = df[mask_row]
    if df.empty: return [], []
    
    header_idx = _find_header_row_index(df)
    header_series = df.iloc[header_idx]
    header = [str(h).strip() for h in header_series.tolist()]
    
    data_part = df.iloc[header_idx + 1 :]
    rows = [[str(c).strip() for c in row] for _, row in data_part.iterrows()]
    rows = [r for r in rows if any(_has_meaningful_text(c) for c in r)]
    
    return header, rows


# -------------------------------
# Main Extraction
# -------------------------------

def extract_tables(
    file_path: str | Path,
    doc_id: str,
    doc_type: str = "generic",  # NOTE: doc_type currently unused but kept for interface consistency
    pages: str = "all",
    flavor_priority: Optional[list[str]] = None,
) -> List[TableBlock]:
    path = Path(file_path)
    if not path.exists(): raise FileNotFoundError(f"PDF not found: {path}")

    llm_client = _get_llm_client()
    
    # Store distinct lists to handle prioritization
    vision_tables: List[TableBlock] = []
    camelot_tables: List[TableBlock] = []
    
    # Hash Set for De-duplication
    seen_content_hashes = set()
    
    global_table_counter = 0
    
    # --- VISION STRATEGY ---
    if llm_client:
        try:
            doc = fitz.open(path)
            page_indices = range(len(doc))
            if pages != "all":
                try:
                    if "-" in pages:
                        start, end = map(int, pages.split("-"))
                        page_indices = range(start-1, end)
                    else:
                        page_indices = [int(p)-1 for p in pages.split(",")]
                except: pass

            for page_idx in page_indices:
                if page_idx >= len(doc): continue
                page = doc[page_idx]
                
                text_instances = []
                keywords = ["ตารางที่", "Table", "ลำดับ", "รายการ"] + [p.replace(r".*","").replace(r".?","") for p in HEADER_PATTERNS]
                for kw in keywords:
                    text_instances.extend(page.search_for(kw))
                
                areas_to_process = []
                if text_instances:
                    min_y = min(r.y0 for r in text_instances)
                    clip_rect = fitz.Rect(0, max(0, min_y - 50), page.rect.width, page.rect.height)
                    areas_to_process.append(clip_rect)
                else:
                    drawings = page.get_drawings()
                    if len(drawings) > 20: areas_to_process.append(page.rect)

                for clip_rect in areas_to_process:
                    pix = page.get_pixmap(clip=clip_rect, matrix=fitz.Matrix(2, 2))
                    img_data = pix.tobytes("png")
                    image = Image.open(io.BytesIO(img_data))
                    
                    print(f"[table_extractor] Vision processing page {page_idx+1}...")
                    html_content = _extract_table_with_vision(llm_client, image)
                    if not html_content: continue

                    columns, rows, has_complex_body, has_complex_header = parse_html_table(html_content)
                    
                    # [CHANGE] Filter Junk Tables (Header only, No data rows)
                    if columns and not rows:
                        print(f"[table_extractor] Dropping junk Vision table (Only header found) on page {page_idx+1}")
                        continue

                    # [CHANGE] Content Hash Check (Row Content Only)
                    if rows:
                        content_hash = _compute_row_content_hash(rows)
                        if content_hash in seen_content_hashes:
                             print(f"[table_extractor] Skipping duplicate Vision table on page {page_idx+1}")
                             continue
                        seen_content_hashes.add(content_hash)

                    # ... (Logic เดิมส่วน Metadata) ...
                    summary_text = _summarize_table(llm_client, html_content, is_html=True)
                    # FIX TASK 6: Robust Classification Input
                    header_hints = _extract_text_from_html_headers(html_content)
                    row_hint = " ".join(rows[0]) if rows else ""
                    sample_for_classify = f"{summary_text} {header_hints} {row_hint}".strip()
                        
                    category = _classify_category_with_llm(llm_client, sample_for_classify)
                    
                    global_table_counter += 1
                    table_id = f"tbl_{doc_id}_{page_idx+1:03d}_{global_table_counter:04d}"
                    
                    # Markdown construction
                    if html_content:
                        markdown_content = html_content
                    else:
                        markdown_content = ""

                    structured_available = False
                    raw_available = False
                    structure_lossy = True
                    markdown_is_html = True
                    
                    if columns and rows and not has_complex_body:
                         markdown_content = table_to_markdown(columns, rows)
                         structured_available = True
                         structure_lossy = False
                         markdown_is_html = False

                    vision_tables.append(TableBlock(
                        id=table_id,
                        doc_id=doc_id,
                        page=page_idx + 1,
                        name=f"Table {global_table_counter} (Vision)",
                        section=None,
                        category=category,
                        columns=columns,
                        rows=rows,
                        markdown=markdown_content,
                        bbox=(clip_rect.x0, clip_rect.y0, clip_rect.x1, clip_rect.y1),
                        extra={
                            "html_content": html_content,
                            "summary": summary_text,
                            "method": "qwen_vision",
                            "structured_available": structured_available,
                            "raw_available": raw_available,
                            "structure_lossy": structure_lossy,
                            "markdown_is_html": markdown_is_html,
                            "source": "vision",
                            "role": category,
                            "numeric_trust": "low",
                            "schema_version": SCHEMA_VERSION, 
                            "parser_version": PARSER_VERSION
                        },
                    ))
                    

        except Exception as e:
            print(f"[table_extractor] Vision failed: {e}. Fallback...")

    # --- CAMELOT STRATEGY (Fallback) ---
    print("[table_extractor] Using Camelot...")
    if flavor_priority is None: flavor_priority = ["lattice", "stream"]

    for flavor in flavor_priority:
        try:
            tables = camelot.read_pdf(str(path), pages=pages, flavor=flavor)
        except: continue
        if tables.n == 0: continue

        for t in tables:
            columns, rows = _dataframe_to_columns_rows(t.df)
            if len(columns) < MIN_COLS: continue

            sub_tables = _split_rows_by_header(rows)
            items_to_add = sub_tables if sub_tables else [("Table", rows)]
            
            for header_txt, sub_rows in items_to_add:
                # [CHANGE] Add Junk Filter (Rows <= 1 OR Cols <= 1 OR Specific Junk Text)
                if len(sub_rows) <= 1 or len(columns) <= 1: 
                    print(f"[table_extractor] Dropping junk Camelot table (Size too small) on page {t.page}")
                    continue
                
                # Filter specific junk text (e.g. Table 5)
                row_text_combined = "".join(["".join(r) for r in sub_rows])
                if "อยากให้ลงชื่อ" in row_text_combined:
                    print(f"[table_extractor] Dropping junk Camelot table (Junk text detected) on page {t.page}")
                    continue
                
                # [CHANGE] Add De-duplication using Hash (Row Content Only)
                # Check against seen_content_hashes to prevent duplicates from flavor/vision
                content_hash = _compute_row_content_hash(sub_rows)
                if content_hash in seen_content_hashes:
                    print(f"[table_extractor] Skipping duplicate Camelot table on page {t.page}")
                    continue
                seen_content_hashes.add(content_hash)

                global_table_counter += 1
                
                markdown = table_to_markdown(columns, sub_rows)
                html = table_to_html(columns, sub_rows)
                
                summary = ""
                cat = "generic_table"
                
                if llm_client:
                    summary = _summarize_table(llm_client, markdown, is_html=False)
                    
                    cat = _classify_category_with_llm(llm_client, f"{header_txt} {' '.join(columns)}")
                
                table_id = f"tbl_{doc_id}_{t.page:03d}_{global_table_counter:04d}"
                
                structured = True
                if not columns or not sub_rows: structured = False

                camelot_tables.append(TableBlock(
                    id=table_id,
                    doc_id=doc_id,
                    page=t.page,
                    name=header_txt,
                    section=None,
                    category=cat,
                    columns=columns,
                    rows=sub_rows,
                    markdown=markdown,
                    bbox=None,
                    extra={
                        "html_content": html,
                        "summary": summary,
                        "method": "camelot",
                        "structured_available": structured,
                        "raw_available": True, # Camelot trusted for math
                        "structure_lossy": False,
                        "markdown_is_html": False,
                        "source": "camelot",
                        "role": cat,
                        "numeric_trust": "high", 
                        "schema_version": SCHEMA_VERSION,
                        "parser_version": PARSER_VERSION
                    }
                ))
        
    # [CHANGE] Final Conflict Resolution Logic (Per Page)
    # ถ้าหน้าไหน Camelot อ่านเจอ (ซึ่งแม่นยำกว่าเรื่อง Text) ให้ทิ้ง Vision ในหน้านั้นไปเลย
    # เพื่อป้องกันข้อมูลซ้ำซ้อนและ Vision Hallucination
    
    final_tables = []
    # สร้าง Set เก็บเลขหน้าที่ Camelot ทำงานได้
    pages_with_camelot = set(t.page for t in camelot_tables)
    
    # 1. เพิ่มตาราง Camelot ทั้งหมด (เพราะเชื่อถือได้สุด)
    final_tables.extend(camelot_tables)
    
    # 2. เพิ่มตาราง Vision เฉพาะหน้าที่ Camelot *หาไม่เจอ*
    for v_table in vision_tables:
        if v_table.page not in pages_with_camelot:
            final_tables.append(v_table)
        else:
            print(f"[table_extractor] Dropping Vision table on page {v_table.page} because Camelot succeeded.")
            
    # เรียงลำดับตามหน้าและ ID เพื่อความสวยงาม
    final_tables.sort(key=lambda x: (x.page, x.id))
    
    return final_tables


def table_to_text(table: TableBlock) -> str:
    lines = []
    if table.name: lines.append(f"ตาราง: {table.name}")
    if table.extra.get("summary"): lines.append(f"สรุป: {table.extra['summary']}")
    
    if hasattr(table, "markdown") and table.markdown:
        lines.append(table.markdown)
    else:
        if table.columns: lines.append(" | ".join(map(str, table.columns)))
        for r in table.rows[:5]:
            lines.append(" | ".join(map(str, r)))
            
    return "\n".join(lines)


def compute_from_table(table: TableBlock, operation: str = "sum", column: str = None):
    # FIX TASK 1: Prevent calc on vision tables even if structured available
    extra = getattr(table, "extra", {}) or {}
    if not extra.get("raw_available", False) or extra.get("numeric_trust") == "low":
        # TODO: Implement fuzzy calc or return warning for low trust tables
        raise ValueError("Cannot compute: Vision-based table data is not trustable for calculation.")

    import pandas as pd
    try:
        df = pd.DataFrame(table.rows, columns=table.columns)
    except: raise ValueError("DF creation failed")
    
    if column is None:
        for c in df.columns:
            try:
                df[c] = pd.to_numeric(df[c])
                column = c; break
            except: pass
    
    if not column: raise ValueError("No numeric column")
    
    if operation == "sum": return df[column].sum()
    elif operation == "mean": return df[column].mean()
    elif operation == "max": return df[column].max()
    return "Unknown Operation"