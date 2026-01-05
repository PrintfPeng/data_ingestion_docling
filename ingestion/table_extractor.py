from __future__ import annotations

"""
table_extractor.py

หน้าที่:
- ใช้ Camelot อ่านตารางจาก PDF (วิธีเดิม)
- [NEW] ใช้ Gemini Vision อ่านตารางจากภาพ (วิธีใหม่สำหรับตารางซับซ้อน)
- [NEW] ทำความสะอาดภาษาไทย (ลบ \\n กลางคำ)
- [NEW] AI-Powered Classification: ใช้ LLM จำแนกหมวดหมู่ตารางแทน Regex
- แปลงตารางเป็น HTML (Display) และ Markdown (AI Context)
- คืนค่า list[TableBlock]
"""

import time
import io
import re  # เพิ่ม regex สำหรับ clean text
from pathlib import Path
from typing import List, Optional, Any, Tuple

import camelot
import pandas as pd
import fitz  # PyMuPDF
from PIL import Image
import google.generativeai as genai

from .schema import TableBlock, BBox
from .config import GOOGLE_API_KEY
from dotenv import load_dotenv
import os

load_dotenv()
# -------------------------------
# Config / Heuristics
# -------------------------------

MIN_ROWS = 2
MIN_COLS = 2
MAX_HEADER_SCAN_ROWS = 3

# [NEW] Pattern สำหรับตรวจจับ header ของกลุ่มย่อยในตาราง (Still useful for splitting)
HEADER_PATTERNS = [
    r"ถือป.?ายสโลแกน",   # กลุ่มถือป้าย
    r"ขบวนสาด",          # ขบวนแห่
    r"แฟนซี",            # แฟนซี
    r"📌.*ถือป.?าย",     # รูปแบบที่มี emoji
    r"📌.*ขบวน",
    r"📌.*แฟนซี",
    r"สว\.?",            # สภานักเรียน/สว.
    r"คนทำอุปกรณ์",
]

# [NEW] ฟังก์ชันทำความสะอาดข้อความภาษาไทย
def _clean_thai_text(text: Any) -> str:
    if text is None:
        return ""
    text = str(text).strip()
    
    # 1. ลบ \n ที่อยู่ระหว่างตัวอักษรไทย (เชื่อมคำที่ถูกตัดบรรทัด)
    # Pattern: ตัวไทย -> \n -> ตัวไทย ==> ลบ \n ทิ้ง
    # \u0E00-\u0E7F คือช่วง Unicode ภาษาไทย
    text = re.sub(r'(?<=[\u0E00-\u0E7F])\s*[\r\n]+\s*(?=[\u0E00-\u0E7F])', '', text)
    
    # 2. ลบ \n ที่เหลือเปลี่ยนเป็น space (สำหรับภาษาอังกฤษหรือเว้นวรรคปกติ)
    text = re.sub(r'[\r\n]+', ' ', text)
    
    # 3. ลบ space ซ้ำซ้อน
    text = re.sub(r'\s+', ' ', text)
    
    # 4. ลบตัวอักษรขยะที่อาจติดมา (เช่น จุดไข่ปลาเยอะๆ)
    text = re.sub(r'\.{3,}', '', text)
    
    return text.strip()


def _has_meaningful_text(s: str) -> bool:
    if s is None:
        return False
    s = str(s).strip()
    if not s:
        return False
    # ต้องมีตัวหนังสือหรือตัวเลขอย่างน้อย 1 ตัว
    return any(ch.isalnum() for ch in s)


# -------------------------------
# [NEW] ฟังก์ชันแยกตารางเป็นกลุ่มย่อยตาม header patterns
# -------------------------------
def _split_rows_by_header(rows: list[list[Any]]) -> list[tuple[str, list[list[Any]]]]:
    """
    แยก rows ออกเป็นกลุ่มย่อยตาม header patterns
    Returns: [(header_text, sub_rows), ...]
    """
    blocks = []
    current_header = "Generic Section" # Default header
    current_rows = []
    
    for r in rows:
        # ตรวจ cell แรกว่าตรงกับ pattern ไหม (หรือ cell ใดๆ ในแถวแรกๆ)
        # บางที header อาจจะอยู่ col 0 หรือ col 1
        row_text = " ".join(str(c) for c in r)
        
        found_header = None
        for p in HEADER_PATTERNS:
            if re.search(p, row_text, re.IGNORECASE):
                found_header = row_text # ใช้ทั้งแถวเป็น header ไปเลยเพื่อความครบถ้วน
                break
        
        if found_header:
            # เจอ header ใหม่ -> บันทึกกลุ่มเก่า
            if current_rows:
                blocks.append((current_header, current_rows))
            
            # เริ่มกลุ่มใหม่
            current_header = _clean_thai_text(found_header) # Clean header ด้วย
            current_rows = [] 
            # (ไม่ add row นี้เข้า current_rows เพราะเป็น header)
        else:
            # แถวข้อมูลธรรมดา
            # Clean ข้อมูลในแถว
            cleaned_row = [_clean_thai_text(c) for c in r]
            
            # เช็คว่าแถวนี้มีข้อมูลที่มีความหมายไหม (ไม่ใช่ว่างเปล่า)
            if any(_has_meaningful_text(c) for c in cleaned_row):
                current_rows.append(cleaned_row)
    
    # บันทึกกลุ่มสุดท้าย
    if current_rows:
        blocks.append((current_header, current_rows))
    
    return blocks


# -------------------------------
# Helper: Gemini Table Summarizer & Vision Extractor & Classifier
# -------------------------------

def _get_gemini_model(model_name="gemini-2.5-flash"): # ใช้ 1.5-flash แทน 2.0 (ถ้ายังไม่มา)
    if not GOOGLE_API_KEY:
        return None
    try:
        genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
        return genai.GenerativeModel(model_name)
    except Exception as e:
        print(f"[table_extractor] Failed to init Gemini: {e}")
        return None

def _summarize_table(model, markdown_table: str) -> str:
    if not model: return ""
    truncated_table = markdown_table[:4000]
    prompt = (
        "คุณเป็นผู้ช่วยวิเคราะห์ข้อมูล จงสรุปใจความสำคัญของตารางนี้สั้นๆ (ไม่เกิน 3 บรรทัด):\n"
        "1. ตารางนี้แสดงข้อมูลเกี่ยวกับอะไร\n"
        "2. มีแนวโน้มหรือตัวเลขที่น่าสนใจอะไรบ้าง (ถ้ามี)\n"
        "ตอบเป็นภาษาไทย กระชับ เข้าใจง่าย\n\n"
        f"ข้อมูลตาราง (Markdown):\n{truncated_table}"
    )
    try:
        response = model.generate_content(prompt)
        return response.text.strip()
    except Exception as e:
        print(f"[table_extractor] Summarization failed: {e}")
        return ""

# [NEW] AI Classification Function
def _classify_category_with_llm(model, text_sample: str) -> str:
    """
    ใช้ AI จำแนกหมวดหมู่ของตารางจากข้อความตัวอย่าง (Header/Row)
    คืนค่าเป็น English Keyword (Snake Case) เพื่อใช้ในการ Filter
    """
    if not model: return "generic_table"
    
    prompt = (
        "คุณคือระบบจำแนกข้อมูลตาราง (Data Classifier) สำหรับงานกิจกรรมกีฬาสีและงานโรงเรียน\n"
        f"จงวิเคราะห์ข้อความ Header หรือ Sample Data ต่อไปนี้: '{text_sample}'\n\n"
        "ให้ระบุว่าตารางนี้น่าจะเกี่ยวกับอะไร โดยเลือกตอบเป็นภาษาอังกฤษ (snake_case) คำเดียว จากตัวเลือกต่อไปนี้ (หรือสร้างใหม่ถ้าจำเป็น):\n"
        "- slogan_holder (คนถือป้าย, สโลแกน, ป้ายคณะสี)\n"
        "- parade (ขบวนแห่, ขบวนพาเหรด, คนเดินขบวน)\n"
        "- fancy (ชุดแฟนซี, การแต่งกายแฟนซี)\n"
        "- student_council (สภานักเรียน, คณะกรรมการ, สว.)\n"
        "- equipment (อุปกรณ์, พัสดุ, คนทำอุปกรณ์)\n"
        "- budget (งบประมาณ, การเงิน, บัญชี)\n"
        "- schedule (กำหนดการ, เวลา)\n"
        "- staff (คณะทำงาน, สตาฟ)\n"
        "- generic_table (ถ้าไม่แน่ใจ หรือเป็นตารางทั่วไป)\n"
        "\nตอบเฉพาะคำศัพท์ (เช่น slogan_holder) ห้ามมีคำอธิบายอื่น"
    )
    try:
        # Config temperature=0.0 เพื่อความแม่นยำและคงที่
        response = model.generate_content(
            prompt, 
            generation_config=genai.types.GenerationConfig(temperature=0.0)
        )
        category = response.text.strip().lower()
        # Clean response เผื่อ AI ตอบยาวเกิน (เอาเฉพาะตัวอักษรและ underscore)
        category = re.sub(r"[^a-z_]", "", category)
        return category if category else "generic_table"
    except Exception as e:
        print(f"[table_extractor] Classification failed: {e}")
        return "generic_table"

# [NEW] Vision Extraction Logic
def _extract_table_with_vision(model, image: Image.Image) -> str:
    """ส่งรูปภาพตารางให้ Gemini Vision แปลงเป็น HTML"""
    prompt = """
    Task: Extract data from the table in this image into an HTML Table.
    
    Rules:
    1. Output ONLY the HTML `<table>...</table>` code. No markdown formatting, no descriptions.
    2. Preserve all Thai characters correctly. Do not break words.
    3. If there are checkmarks (✓), write "Yes" or use a check icon.
    4. Use these CSS classes: <table class="min-w-full text-sm text-left text-slate-600 border-collapse border border-slate-200">
    5. Merge cells (rowspan/colspan) exactly as seen in the image.
    6. Ensure <thead> and <tbody> structure is correct.
    """
    try:
        response = model.generate_content([prompt, image])
        # ล้าง Markdown code block ออก
        html = response.text.replace("```html", "").replace("```", "").strip()
        
        # [Safety Check] ถ้า HTML ไม่สมบูรณ์ ให้คืนค่าว่าง
        if "<table" not in html or "</table>" not in html:
            return ""
            
        return html
    except Exception as e:
        print(f"[table_extractor] Vision extraction failed: {e}")
        return ""


# -------------------------------
# Category detection & DataFrame helpers (Logic เดิม - Deprecated but kept as fallback)
# -------------------------------

def _guess_table_category(columns: list[str], first_row: list[Any]) -> str:
    # [UPDATED] ฟังก์ชันนี้อาจจะไม่ได้ใช้แล้วเพราะมี _classify_category_with_llm แต่เก็บไว้เป็น fallback
    header_lower = " ".join(columns).lower()
    # กันกรณี first_row ว่าง
    first_row_txt = " ".join(str(x) for x in first_row).lower() if first_row else ""
    text_for_detect = header_lower + " " + first_row_txt

    if any(k in text_for_detect for k in ["date", "วันที่", "amount", "ยอด", "เงิน", "คงเหลือ"]):
        return "transaction_table"
    if any(k in text_for_detect for k in ["item", "รายละเอียด", "รายการ", "สินค้า"]):
        return "item_list"
    return "generic_table"

def _find_header_row_index(df: pd.DataFrame) -> int:
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
    if df.empty: return [], []
    
    # [CRITICAL STEP] Clean Thai Text ก่อนเริ่ม process
    # ใช้ map เพื่อ clean ทุก cell ใน DataFrame
    df = df.applymap(_clean_thai_text)
    
    df_str = df.astype(str)
    
    mask_non_empty_row = df_str.apply(lambda r: any(_has_meaningful_text(c) for c in r), axis=1)
    df_str = df_str[mask_non_empty_row]
    if df_str.empty: return [], []
    
    mask_non_empty_col = df_str.apply(lambda c: any(_has_meaningful_text(v) for v in c))
    df_str = df_str.loc[:, mask_non_empty_col]
    if df_str.empty: return [], []

    header_idx = _find_header_row_index(df_str)
    header_series = df_str.iloc[header_idx]
    header = [str(h).strip() for h in header_series.tolist()]

    data_part = df_str.iloc[header_idx + 1 :]
    rows = [[str(c).strip() for c in row] for _, row in data_part.iterrows()]
    rows = [r for r in rows if any(_has_meaningful_text(c) for c in r)]

    return header, rows


# -------------------------------
# Main extraction (Modified to support Vision + Clean Text + AI Classify)
# -------------------------------

def extract_tables(
    file_path: str | Path,
    doc_id: str,
    doc_type: str = "generic",
    pages: str = "all",
    flavor_priority: Optional[list[str]] = None,
) -> List[TableBlock]:
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"PDF file not found: {path}")


    gemini_vision = _get_gemini_model("gemini-2.5-flash") 
    
    all_tables: List[TableBlock] = []
    
    # -------------------------------------------------------------
    # STRATEGY 1: Vision Extraction (Priority)
    # เราจะลองใช้วิธีนี้เป็นหลักถ้า Model พร้อม เพราะแม่นยำกว่ากับตารางไทย
    # -------------------------------------------------------------
    if gemini_vision:
        try:
            doc = fitz.open(path)
            
            # แปลง pages string เป็น list of int
            page_indices = range(len(doc))
            if pages != "all":
                try:
                    # รองรับ "1,2,3" หรือ "1-3" แบบง่ายๆ
                    if "-" in pages:
                        start, end = map(int, pages.split("-"))
                        page_indices = range(start-1, end)
                    else:
                        page_indices = [int(p)-1 for p in pages.split(",")]
                except:
                    pass

            table_counter = 0
            for page_idx in page_indices:
                if page_idx >= len(doc): continue
                page = doc[page_idx]
                
                # [IMPROVED] หาพื้นที่ตารางให้แม่นยำขึ้น
                # ค้นหาคำว่า "ตารางที่" หรือ "Table" เพื่อระบุตำแหน่ง
                # หรือคำที่เป็น Header pattern ของเรา
                text_instances = []
                keywords = ["ตารางที่", "Table", "ลำดับ", "รายการ", "ชื่อ-สกุล"] + [p.replace(r".*","").replace(r".?","") for p in HEADER_PATTERNS]
                
                for kw in keywords:
                    text_instances.extend(page.search_for(kw))
                
                areas_to_process = []
                if text_instances:
                    # รวมกลุ่ม rect ที่ใกล้กัน
                    # (logic อย่างง่าย: ถ้าเจอคำสำคัญ ให้ crop ตั้งแต่จุดนั้นลงไป)
                    # เลือกจุดที่อยู่บนสุด
                    min_y = min(r.y0 for r in text_instances)
                    # Crop พื้นที่ตาราง (กะเอาจากตำแหน่งคำสำคัญลงไปด้านล่าง)
                    # เพิ่มความสูงเป็น 800 หรือจนสุดหน้า
                    clip_rect = fitz.Rect(0, max(0, min_y - 50), page.rect.width, page.rect.height)
                    areas_to_process.append(clip_rect)
                else:
                    # ถ้าไม่เจอคำว่าตาราง ลองสุ่มตรวจหาเส้น (Drawing)
                    # ถ้ามีเส้นเยอะๆ น่าจะมีตาราง
                    drawings = page.get_drawings()
                    if len(drawings) > 20: 
                        areas_to_process.append(page.rect) # ส่งทั้งหน้า

                for clip_rect in areas_to_process:
                    # แปลงเป็นรูป
                    pix = page.get_pixmap(clip=clip_rect, matrix=fitz.Matrix(2, 2)) # Zoom 2x
                    img_data = pix.tobytes("png")
                    image = Image.open(io.BytesIO(img_data))
                    
                    print(f"[table_extractor] Vision processing page {page_idx+1}...")
                    html_content = _extract_table_with_vision(gemini_vision, image)
                    
                    # เช็คว่าได้ HTML Table กลับมาจริงๆ ไหม
                    if not html_content or "<table" not in html_content:
                        continue

                    # สร้าง Summary จาก HTML ที่แกะได้
                    summary_text = _summarize_table(gemini_vision, html_content)
                    
                    # [CRITICAL] ใช้ AI Classify หมวดหมู่
                    # สำหรับ Vision เราอาจจะยังไม่มี Header Text ที่ชัดเจน
                    # เราจะใช้ summary_text หรือ html_content บางส่วนส่งไป Classify
                    sample_for_classify = summary_text if summary_text else html_content[:500]
                    category = _classify_category_with_llm(gemini_vision, sample_for_classify)
                    
                    table_counter += 1
                    table_id = f"tbl_{page_idx+1}_{table_counter:02d}"
                    
                    # สร้าง TableBlock แบบ Dummy (เพราะเราใช้ HTML เป็นหลักแล้ว)
                    all_tables.append(TableBlock(
                        id=table_id,
                        doc_id=doc_id,
                        page=page_idx + 1,
                        name=f"Table {table_counter}",
                        section=None,
                        category=category, # ใช้ AI Category
                        columns=["AI_Extracted"],
                        rows=[["Data in HTML"]],
                        bbox=(clip_rect.x0, clip_rect.y0, clip_rect.x1, clip_rect.y1),
                        extra={
                            "html_content": html_content,
                            "markdown_content": html_content, # ใช้ HTML แทน Markdown ไปเลย
                            "summary": summary_text,
                            "method": "gemini_vision",
                            "role": category # เก็บไว้ใช้ filter
                        },
                    ))
                    time.sleep(1) # Rate limit guard

            # ถ้าเจอด้วย Vision แล้ว ให้ Return เลย (ไม่ต้องใช้ Camelot ต่อ)
            if all_tables:
                return all_tables

        except Exception as e:
            print(f"[table_extractor] Vision strategy failed: {e}, falling back to Camelot...")

    # -------------------------------------------------------------
    # STRATEGY 2: Camelot Extraction (Fallback - วิธีเดิม)
    # -------------------------------------------------------------
    print("[table_extractor] Falling back to Camelot...")
    if flavor_priority is None:
        flavor_priority = ["lattice", "stream"]
    
    gemini_model = _get_gemini_model() # สำหรับ Summarize เฉยๆ
    table_index = 0

    for flavor in flavor_priority:
        try:
            tables = camelot.read_pdf(str(path), pages=pages, flavor=flavor)
        except Exception as e:
            continue

        if tables.n == 0: continue

        for t in tables:
            df: pd.DataFrame = t.df
            
            # [CRITICAL] Clean Text ที่นี่ด้วย!
            columns, rows = _dataframe_to_columns_rows(df)
            
            if len(columns) < MIN_COLS or len(rows) < MIN_ROWS: continue

            # [NEW] ตรวจสอบว่ามี header pattern ที่ต้องแยกกลุ่มย่อยไหม
            sub_tables = _split_rows_by_header(rows)
            
            if sub_tables:
                # กรณีเจอ header patterns -> แยกเป็นตารางย่อย
                print(f"[table_extractor] Found {len(sub_tables)} sub-tables in table {table_index+1}")
                
                for header_text, sub_rows in sub_tables:
                    if len(sub_rows) < 1: # ยอมรับแม้มีแค่ 1 แถวถ้ามี header ชัดเจน
                        continue
                    
                    table_index += 1
                    
                    # สร้าง DataFrame สำหรับตารางย่อย
                    # ใช้ column เดิมถ้าจำนวนตรงกัน ไม่งั้นสร้างใหม่
                    try:
                        if len(columns) == len(sub_rows[0]):
                            clean_df = pd.DataFrame(sub_rows, columns=columns)
                        else:
                            clean_df = pd.DataFrame(sub_rows)
                    except:
                        clean_df = pd.DataFrame(sub_rows)
                    
                    # Generate HTML & Markdown (ใช้ logic เดิม)
                    markdown_content = clean_df.to_markdown(index=False)
                    html_content = clean_df.to_html(index=False, classes="min-w-full text-sm text-left text-slate-600 border-collapse border border-slate-200", border=0)
                    
                    # แต่ง CSS
                    html_content = html_content.replace('<thead>', '<thead class="bg-slate-100 text-slate-700 font-semibold">')
                    html_content = html_content.replace('<th>', '<th class="px-4 py-2 border border-slate-200">')
                    html_content = html_content.replace('<td>', '<td class="px-4 py-2 border border-slate-200 align-top">')
                    
                    # Generate Summary
                    summary_text = ""
                    if gemini_model:
                        summary_text = _summarize_table(gemini_model, markdown_content)
                        time.sleep(1)
                    
                    # [CRITICAL UPDATE] ใช้ AI Classify หมวดหมู่แทนการ Hardcode
                    # ส่ง Header + แถวแรกไปให้ AI ดูบริบท
                    sample_text = f"Header: {header_text}\nData: {' '.join(map(str, sub_rows[0]))}"
                    category = _classify_category_with_llm(gemini_model, sample_text)
                    print(f"[table_extractor] Sub-Table {table_index} classified as: {category}")
                    
                    if gemini_model:
                        time.sleep(1) # Rate limit guard รวมกันตรงนี้
                    
                    all_tables.append(TableBlock(
                        id=f"tbl_{table_index:04d}",
                        doc_id=doc_id,
                        page=t.page,
                        name=header_text,  # ใช้ชื่อ header เป็นชื่อตาราง
                        section=None,
                        category=category, # ใช้ AI Category
                        columns=clean_df.columns.tolist(),
                        rows=sub_rows,
                        bbox=None,
                        extra={
                            "html_content": html_content,
                            "markdown_content": markdown_content,
                            "summary": summary_text,
                            "method": "camelot",
                            "sub_table_header": header_text,
                            # เพิ่ม metadata เพื่อช่วย RAG filter
                            "role": category # เก็บไว้ใช้ filter
                        },
                    ))
            else:
                # กรณีไม่เจอ header patterns -> ใช้โค้ดเดิม (ตารางเดียว)
                try:
                    clean_df = pd.DataFrame(rows, columns=columns)
                except:
                    clean_df = df
                
                markdown_content = clean_df.to_markdown(index=False)
                html_content = clean_df.to_html(index=False, classes="min-w-full text-sm text-left text-slate-600 border-collapse border border-slate-200", border=0)
                
                # แต่ง CSS
                html_content = html_content.replace('<thead>', '<thead class="bg-slate-100 text-slate-700 font-semibold">')
                html_content = html_content.replace('<th>', '<th class="px-4 py-2 border border-slate-200">')
                html_content = html_content.replace('<td>', '<td class="px-4 py-2 border border-slate-200 align-top">')

                # Generate Summary
                summary_text = ""
                if gemini_model:
                    summary_text = _summarize_table(gemini_model, markdown_content)
                    # time.sleep(1)

                # [CRITICAL UPDATE] ใช้ AI Classify หมวดหมู่
                # ส่ง Columns + แถวแรกไปให้ AI ดูบริบท
                sample_text = f"Columns: {' '.join(columns)}\nData: {' '.join(map(str, rows[0]))}"
                category = _classify_category_with_llm(gemini_model, sample_text)
                print(f"[table_extractor] Table {table_index+1} classified as: {category}")
                
                if gemini_model:
                    time.sleep(1)

                table_index += 1
                all_tables.append(TableBlock(
                    id=f"tbl_{table_index:04d}",
                    doc_id=doc_id,
                    page=t.page,
                    name=f"table_{table_index}",
                    section=None,
                    category=category, # ใช้ AI Category
                    columns=columns,
                    rows=rows,
                    bbox=None,
                    extra={
                        "html_content": html_content,
                        "markdown_content": markdown_content,
                        "summary": summary_text,
                        "method": "camelot",
                        "role": category # เก็บไว้ใช้ filter
                    },
                ))
        
        if all_tables: break

    return all_tables

# [NEW] Helper to convert TableBlock to text for embedding (RAG)
def table_to_text(table):
    lines = []
    # Check extra safely
    extra = getattr(table, "extra", {}) or {} 
    if extra.get("summary"):
        lines.append(extra["summary"])
    
    # Join columns
    if table.columns:
        lines.append(" | ".join(map(str, table.columns)))
        
    # Join rows (limit 5)
    if table.rows:
        for row in table.rows[:5]:
            lines.append(" | ".join(map(str, row)))
            
    return "\n".join(lines)
