from __future__ import annotations

"""
semantic_enricher.py (Universal Edition + Hybrid AI)

รวมฟังก์ชัน:
1) Section Segmentation / Categorization (รองรับ Finance, Legal, Work, Knowledge)
2) Text Role Categorization (title, transaction, legal_clause, instruction_step ฯลฯ)
3) Table Normalization (header → canonical names)
4) Table Role Categorization
5) Mapping Prepare: ดึงรายการ transaction ออกมา

ทำงานได้ทั้งแบบ:
- rule-based อย่างเดียว (Keyword ครอบคลุมทุกประเภท)
- ใช้ LLM (Custom API/Qwen) เป็นตัวหลัก
- ใช้ Google Gemini เป็นตัวสำรอง (Fallback)
"""

from typing import List, Dict, Any, Optional
from dotenv import load_dotenv
import os
import re

load_dotenv()

# [CHANGE] ใช้ OpenAI Client สำหรับ Custom API
try:
    from openai import OpenAI
except ImportError:
    OpenAI = None

# [NEW] เพิ่ม Google GenAI สำหรับแผนสำรอง
try:
    import google.generativeai as genai
    HAS_GOOGLE = True
except ImportError:
    HAS_GOOGLE = False

from .schema import IngestedDocument, TextBlock, TableBlock

# ---------------------------
# [CHANGE] Model Config
# ---------------------------
LLM_MODEL = os.getenv("CUSTOM_MODEL_NAME", "qwen/qwen-2.5-72b-instruct")


def _get_llm_client() -> Optional[OpenAI]:
    """คืน OpenAI Client สำหรับ Custom API ถ้ามี Key"""
    api_key = os.getenv("CUSTOM_API_KEY")
    base_url = os.getenv("CUSTOM_API_BASE")
    
    if api_key:
        pass
    else:
        # print("[DEBUG semantic_enricher] No CUSTOM_API_KEY found.")
        return None

    try:
        return OpenAI(api_key=api_key, base_url=base_url, timeout=45)
    except Exception as e:
        print("[semantic_enricher] Cannot init OpenAI Client:", e)
        return None

def _get_google_client():
    """คืน Google Client สำหรับ Backup"""
    if not HAS_GOOGLE: return None
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key: return None
    try:
        genai.configure(api_key=api_key)
        return genai.GenerativeModel('gemini-2.5-flash')
    except: return None


# ===========================
# 1) SECTION TAGGING
# ===========================

# [UPDATE] เพิ่ม Section ใหม่ตามประเภทเอกสาร
SECTION_LABELS = [
    "header", 
    "summary", 
    "transactions", 
    "legal_section",      # [NEW] สัญญา/กฎหมาย
    "instruction_section", # [NEW] คู่มือ/ขั้นตอน
    "work_section",       # [NEW] การทำงาน/ประชุม/Resume
    "footer", 
    "qna", 
    "other"
]

_QNA_HINTS = ["ถาม:", "คำถาม", "ข้อที่", "จงตอบ", "เลือกคำตอบ", "question"]

def _looks_like_qna(text: str) -> bool:
    t = text.replace(" ", "")
    return any(h in t for h in _QNA_HINTS)

def _guess_section_rule(block: TextBlock, index: int, total: int) -> str:
    """
    rule-based segmentation (Universal):
    รองรับ Finance, Legal, Work, Knowledge ตาม Keyword ที่ระบุ
    """
    txt = (block.content or "").strip()
    lower = txt.lower()
    extra = block.extra or {}

    is_heading = bool(extra.get("is_heading"))
    page = getattr(block, "page", None) or 0

    # 1. Q&A / Exam
    if _looks_like_qna(txt):
        return "qna"
    if any(k in lower for k in ["exam", "test", "quiz", "ข้อสอบ", "แบบฝึกหัด"]):
        return "qna"

    # 2. Header
    if is_heading and (index < 10 or page <= 2):
        return "header"
    if index == 0 and len(txt) <= 120:
        return "header"

    # 3. Summary
    if any(k in lower for k in ["summary", "สรุป", "overview", "executive summary", "บทคัดย่อ", "abstract"]):
        return "summary"

    # 4. Finance (Transactions)
    if any(k in lower for k in ["รายการเดินบัญชี", "statement", "movement", "invoice", "receipt", "ใบแจ้งหนี้", "ใบเสร็จ", "tax form"]):
        return "transactions"

    # 5. Legal (Contracts / Agreements)
    if any(k in lower for k in ["contract", "agreement", "สัญญา", "ข้อตกลง", "mou", "official", "ประกาศ", "ระเบียบ", "gazette"]):
        return "legal_section"

    # 6. Work (Meeting / Resume)
    if any(k in lower for k in ["resume", "cv", "curriculum vitae", "ประวัติ", "minutes", "บันทึกการประชุม", "agenda", "วาระ"]):
        return "work_section"

    # 7. Knowledge (Manuals / Guides)
    if any(k in lower for k in ["manual", "guide", "handbook", "คู่มือ", "instruction", "methodology", "introduction"]):
        return "instruction_section"

    # 8. Footer
    if any(k in lower for k in ["ลงชื่อ", "ผู้มีอำนาจลงนาม", "ขอแสดงความนับถือ", "signature"]):
        return "footer"

    return "other"


def tag_sections(
    doc: IngestedDocument,
    use_llm: bool = False,
) -> IngestedDocument:
    """
    ใส่ section label ลงใน TextBlock.extra["section"]
    """
    client = _get_llm_client() if use_llm else None
    google_client = _get_google_client() if use_llm else None

    if client or google_client:
        joined = []
        for i, b in enumerate(doc.texts):
            joined.append(f"[{i}] {b.content}")
        prompt_text = "\n".join(joined[:200])

        prompt = f"""
You are a document segmenter. Assign ONE section label from:
{SECTION_LABELS}

- header            : document title / main headings
- summary           : executive summary / overview / abstract
- transactions      : financial items / invoice details / statement rows
- legal_section     : contract terms / clauses / regulations
- instruction_section : user manual / guide steps / procedures
- work_section      : resume details / meeting minutes / agenda
- footer            : signatures / closing
- qna               : exam questions / interview Q&A
- other             : anything else

Format: index: label
Text blocks:
{prompt_text}
"""
        mapping: Dict[int, str] = {}
        success = False

        # --- Plan A: OpenRouter ---
        if client:
            try:
                response = client.chat.completions.create(
                    model=LLM_MODEL,
                    messages=[
                        {"role": "system", "content": "You are a helpful document analyzer."},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.0,
                    max_tokens=2000
                )
                resp_text = response.choices[0].message.content or ""
                success = True
            except Exception as e:
                print(f"[semantic_enricher] Plan A (Section) failed: {e}")
        
        # --- Plan B: Google Gemini ---
        if not success and google_client:
            try:
                print("[semantic_enricher] Switching to Plan B (Google) for Section Tagging...")
                response = google_client.generate_content(prompt)
                resp_text = response.text
                success = True
            except Exception as e:
                print(f"[semantic_enricher] Plan B (Section) failed: {e}")

        if success:
            try:
                for line in resp_text.splitlines():
                    line = line.strip()
                    if not line or ":" not in line: continue
                    idx_str, label = line.split(":", 1)
                    idx_str = idx_str.strip().strip("[]")
                    label = label.strip().lower()
                    try: idx = int(idx_str)
                    except: continue
                    if label not in SECTION_LABELS: label = "other"
                    mapping[idx] = label

                total = len(doc.texts)
                for i, b in enumerate(doc.texts):
                    extra = dict(b.extra or {})
                    extra["section"] = mapping.get(i, _guess_section_rule(b, i, total))
                    b.extra = extra
                return doc
            except Exception as e:
                print("[semantic_enricher] Parsing AI response failed:", e)

    # Fallback Rule-based
    print("[semantic_enricher] Using Rule-based Fallback for Section Tagging")
    total = len(doc.texts)
    for i, b in enumerate(doc.texts):
        extra = dict(b.extra or {})
        extra["section"] = _guess_section_rule(b, i, total)
        b.extra = extra

    return doc


# ===========================
# 2) TEXT ROLE CATEGORIZATION
# ===========================

# [UPDATE] เพิ่ม Role ใหม่
TEXT_ROLE_LABELS = [
    "title", 
    "account_info", 
    "transaction_header", 
    "transaction_row", 
    "legal_clause",       # [NEW] ข้อกฎหมาย
    "instruction_step",   # [NEW] ขั้นตอนปฏิบัติ
    "warning_note",       # [NEW] คำเตือน
    "note", 
    "footer_text", 
    "qna_question", 
    "qna_answer", 
    "other"
]

def _guess_text_role_rule(block: TextBlock) -> str:
    """
    Rule-based role guessing (Universal)
    """
    txt = (block.content or "").strip()
    lower = txt.lower()
    extra = block.extra or {}
    section = extra.get("section")
    is_heading = extra.get("is_heading", False)

    # 1. Q&A
    if txt.replace(" ", "").startswith("ถาม:") or "คำถาม" in txt or "question" in lower: return "qna_question"
    if txt.replace(" ", "").startswith("ตอบ:") or "เฉลย" in txt or "answer" in lower: return "qna_answer"
    if "?" in txt and len(txt) < 200: return "qna_question"

    # 2. Title
    if section == "header" and (is_heading or len(txt) < 120): return "title"

    # 3. Finance Roles
    if any(k in lower for k in ["เลขที่บัญชี", "account no", "bank"]): return "account_info"
    if any(k in lower for k in ["วันที่", "date", "amount", "balance", "คงเหลือ"]): return "transaction_header"
    if section == "transactions" and 10 <= len(txt) <= 200: return "transaction_row"

    # 4. Legal Roles (Contract Clauses)
    if any(k in lower for k in ["ข้อที่", "article", "clause", "section", "มาตรา"]): return "legal_clause"
    if section == "legal_section" and re.match(r"^\d+\.", txt): return "legal_clause"

    # 5. Knowledge Roles (Manuals/Guides)
    if any(k in lower for k in ["คำเตือน", "warning", "caution", "ข้อควรระวัง", "note:", "หมายเหตุ"]): return "warning_note"
    if any(k in lower for k in ["ขั้นตอนที่", "step", "method", "วิธีทำ"]): return "instruction_step"

    # 6. Note/Footer
    if any(k in lower for k in ["หมายเหตุ", "remark"]): return "note"
    if any(k in lower for k in ["ลงชื่อ", "signature"]): return "footer_text"

    return "other"


def categorize_text_blocks(
    doc: IngestedDocument,
    use_llm: bool = False,
) -> IngestedDocument:
    """
    ใส่ role ให้ TextBlock.extra["role"]
    """
    client = _get_llm_client() if use_llm else None
    google_client = _get_google_client() if use_llm else None

    if client or google_client:
        joined = []
        for i, b in enumerate(doc.texts[:200]):
            section = (b.extra or {}).get("section", "unknown")
            joined.append(f"[{i}] (section={section}) {b.content}")
        prompt_text = "\n".join(joined)

        prompt = f"""
You are a universal document role classifier. Assign ONE role from:
{TEXT_ROLE_LABELS}

- title             : main headings
- account_info      : bank account / party details
- transaction_header: table headers (date, amount)
- transaction_row   : list item row / transaction line
- legal_clause      : contract terms / articles / regulations
- instruction_step  : manual steps / procedures
- warning_note      : warnings / cautions / important notes
- footer_text       : signatures / closing
- qna_question      : exam question
- qna_answer        : answer key
- other             : normal paragraph

Format: index: role
Text blocks:
{prompt_text}
"""
        mapping: Dict[int, str] = {}
        success = False

        # --- Plan A: OpenRouter ---
        if client:
            try:
                response = client.chat.completions.create(
                    model=LLM_MODEL,
                    messages=[
                        {"role": "system", "content": "You are a helpful document analyzer."},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.0,
                    max_tokens=2000
                )
                resp_text = response.choices[0].message.content or ""
                success = True
            except Exception as e:
                print(f"[semantic_enricher] Plan A (Role) failed: {e}")

        # --- Plan B: Google Gemini ---
        if not success and google_client:
            try:
                print("[semantic_enricher] Switching to Plan B (Google) for Role Tagging...")
                response = google_client.generate_content(prompt)
                resp_text = response.text
                success = True
            except Exception as e:
                print(f"[semantic_enricher] Plan B (Role) failed: {e}")

        if success:
            try:
                for line in resp_text.splitlines():
                    line = line.strip()
                    if not line or ":" not in line: continue
                    idx_str, label = line.split(":", 1)
                    idx_str = idx_str.strip().strip("[]")
                    label = label.strip().lower()
                    try: idx = int(idx_str)
                    except: continue
                    if label not in TEXT_ROLE_LABELS: label = "other"
                    mapping[idx] = label

                for i, b in enumerate(doc.texts):
                    extra = dict(b.extra or {})
                    extra["role"] = mapping.get(i, _guess_text_role_rule(b))
                    b.extra = extra
                return doc
            except Exception as e:
                print("[semantic_enricher] Parsing AI response failed:", e)

    # Fallback Rule-based
    print("[semantic_enricher] Using Rule-based Fallback for Role Tagging")
    for b in doc.texts:
        extra = dict(b.extra or {})
        extra["role"] = _guess_text_role_rule(b)
        b.extra = extra

    return doc


# ===========================
# 4) TABLE NORMALIZER + ROLE
# ===========================

HEADER_NORMALIZATION_MAP = {
    # Finance
    "date": "date", "วันที่": "date", "วันเดือนปี": "date",
    "description": "description", "รายการ": "description", "details": "description",
    "withdrawal": "amount_out", "debit": "amount_out", "จ่าย": "amount_out",
    "deposit": "amount_in", "credit": "amount_in", "รับ": "amount_in",
    "balance": "balance", "คงเหลือ": "balance",
    "amount": "amount", "จำนวนเงิน": "amount",
    # Inventory/General
    "qty": "quantity", "จำนวน": "quantity",
    "unit": "unit", "หน่วย": "unit",
    "price": "unit_price", "ราคา": "unit_price"
}

def _normalize_header_name(h: str) -> str:
    h_clean = (h or "").strip().lower()
    if not h_clean: return ""
    for key, canonical in HEADER_NORMALIZATION_MAP.items():
        if key in h_clean: return canonical
    return h_clean

TABLE_ROLE_LABELS = ["transaction_table", "summary_table", "other_table"]

def _guess_table_role(tb: TableBlock) -> str:
    header = getattr(tb, "header", []) or []
    header_lower = [str(h).lower() for h in header]
    header_joined = " ".join(header_lower)

    has_date = any("date" in h or "วันที่" in h for h in header_lower)
    has_amount = any(any(x in h for x in ["amount", "ยอด", "debit", "credit", "balance", "price"]) for h in header_lower)

    if has_date and has_amount: return "transaction_table"
    if any(k in header_joined for k in ["summary", "สรุป", "total", "รวม"]): return "summary_table"
    return "other_table"

def normalize_tables(tables: List[TableBlock]) -> List[TableBlock]:
    for tb in tables:
        header = list(getattr(tb, "header", []) or [])
        rows = list(getattr(tb, "rows", []) or [])

        # Infer header from first row if missing
        if not header and rows:
            first = rows[0]
            text_cells = sum(1 for c in first if re.search(r"[A-Za-z\u0E00-\u0E7F]", str(c)))
            if text_cells >= max(1, len(first) // 2):
                header = [str(c) for c in first]
                rows = rows[1:]
                extra = dict(tb.extra or {})
                extra["header_inferred"] = True
                tb.extra = extra

        normalized_header = [_normalize_header_name(h) for h in header]
        tb.header = normalized_header
        tb.rows = rows

        extra = dict(tb.extra or {})
        extra_norm = dict(extra.get("header_normalization", {}))
        extra_norm.update({"original_header": header, "normalized_header": normalized_header})
        extra["header_normalization"] = extra_norm
        extra["role"] = _guess_table_role(tb)
        tb.extra = extra

    return tables


# ===========================
# 5) MAPPING PREPARE (transactions)
# ===========================

def _parse_float_safe(val: Optional[str]) -> Optional[float]:
    if val is None: return None
    s = str(val).strip().replace(",", "").replace("฿", "")
    if s.startswith("(") and s.endswith(")"): s = "-" + s[1:-1]
    s = s.replace(" ", "")
    try: return float(s)
    except ValueError: return None

def extract_transactions_from_table(tb: TableBlock) -> List[Dict[str, Any]]:
    header = getattr(tb, "header", []) or []
    rows = getattr(tb, "rows", []) or []
    name_to_idx = {h: i for i, h in enumerate(header) if h}
    records = []

    for row in rows:
        def col(name: str) -> Optional[str]:
            idx = name_to_idx.get(name)
            if idx is None or idx >= len(row): return None
            return str(row[idx]).strip()

        date = col("date")
        desc = col("description")
        amount_in = col("amount_in")
        amount_out = col("amount_out")
        amount = col("amount")
        balance = col("balance")

        # Fallback for inventory invoices
        if not amount:
            qty = _parse_float_safe(col("quantity"))
            u_price = _parse_float_safe(col("unit_price"))
            if qty and u_price: amount = str(qty * u_price)

        if not any([date, desc, amount_in, amount_out, amount, balance]): continue

        record = {
            "date_raw": date,
            "description": desc,
            "amount_in_raw": amount_in,
            "amount_out_raw": amount_out,
            "amount_raw": amount,
            "balance_raw": balance,
            "amount_in": _parse_float_safe(amount_in),
            "amount_out": _parse_float_safe(amount_out),
            "amount": _parse_float_safe(amount),
            "balance": _parse_float_safe(balance),
        }
        records.append(record)
    return records

def prepare_mapping_payload(doc: IngestedDocument) -> Dict[str, Any]:
    all_transactions = []
    for tb in doc.tables:
        txs = extract_transactions_from_table(tb)
        if txs: all_transactions.extend(txs)

    return {
        "doc_id": doc.metadata.doc_id,
        "doc_type": doc.metadata.doc_type,
        "file_name": doc.metadata.filename,
        "transactions": all_transactions,
    }