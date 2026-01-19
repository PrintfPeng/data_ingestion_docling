from __future__ import annotations

"""
semantic_enricher.py

รวมฟังก์ชัน:
1) Section Segmentation / Categorization สำหรับ TextBlock
2) Text Role Categorization (title, account_info, transaction_row, qna_question ฯลฯ)
3) Table Normalization (header → canonical names)
4) Table Role Categorization (transaction_table / summary_table / other_table)
5) Mapping Prepare: ดึงรายการ transaction ออกมาในรูปแบบโครงสร้าง

ทำงานได้ทั้งแบบ:
- rule-based อย่างเดียว (ถ้าไม่มี KEY)
- ใช้ LLM (Custom API/Qwen) ช่วย (ถ้ามี KEY)
"""

from typing import List, Dict, Any, Optional
from dotenv import load_dotenv
import os

load_dotenv()
import re

# [CHANGE] ใช้ OpenAI Client สำหรับ Custom API
try:
    from openai import OpenAI
except ImportError:
    OpenAI = None

from .schema import IngestedDocument, TextBlock, TableBlock

# ---------------------------
# [CHANGE] Model Config
# ---------------------------
# ใช้ Qwen 72B ซึ่งฉลาดที่สุดในลิสต์สำหรับการเข้าใจบริบท
LLM_MODEL = os.getenv("CUSTOM_MODEL_NAME", "qwen/qwen-2.5-72b-instruct")


def _get_llm_client() -> Optional[OpenAI]:
    """
    คืน OpenAI Client สำหรับ Custom API ถ้ามี Key
    """
    api_key = os.getenv("CUSTOM_API_KEY")
    base_url = os.getenv("CUSTOM_API_BASE")
    
    if api_key:
        print(f"[DEBUG semantic_enricher] Custom API Key found: {api_key[:5]}...")
    else:
        print("[DEBUG semantic_enricher] No CUSTOM_API_KEY found.")
        return None

    try:
        return OpenAI(api_key=api_key, base_url=base_url)
    except Exception as e:
        print("[semantic_enricher] Cannot init OpenAI Client:", e)
        return None


# ===========================
# 1) SECTION TAGGING
# ===========================

SECTION_LABELS = ["header", "summary", "transactions", "footer", "qna", "other"]

_QNA_HINTS = [
    "ถาม:",
    "คำถาม",
    "ข้อที่",
    "จงตอบ",
    "เลือกคำตอบ",
    "question",
]


def _looks_like_qna(text: str) -> bool:
    t = text.replace(" ", "")
    return any(h in t for h in _QNA_HINTS)


def _guess_section_rule(block: TextBlock, index: int, total: int) -> str:
    """
    rule-based segmentation:
    - ใช้ is_heading จาก pdf_parser ถ้ามี
    - ใช้ pattern ภาษาไทย/อังกฤษ ทั่วไป
    """
    txt = (block.content or "").strip()
    lower = txt.lower()
    extra = block.extra or {}

    is_heading = bool(extra.get("is_heading"))
    page = getattr(block, "page", None) or 0

    # 1) Q&A document section
    if _looks_like_qna(txt):
        return "qna"

    # 2) header: หัวเรื่องใหญ่ต้นเอกสาร
    if is_heading and (index < 10 or page <= 2):
        return "header"
    if index == 0 and len(txt) <= 120:
        return "header"

    # 3) summary
    if any(k in lower for k in ["summary", "สรุป", "overview", "executive summary", "สรุปยอด"]):
        return "summary"

    # 4) transaction-like
    if any(k in lower for k in ["รายการเดินบัญชี", "statement of account", "movement"]):
        return "transactions"
    if any(k in lower for k in ["รายการ", "รายละเอียดบัญชี", "statement", "transactions"]):
        return "transactions"

    # 5) footer
    if any(k in lower for k in ["ลงชื่อ", "ผู้มีอำนาจลงนาม", "ขอแสดงความนับถือ", "signature"]):
        return "footer"

    return "other"


def tag_sections(
    doc: IngestedDocument,
    use_llm: bool = False,
) -> IngestedDocument:
    """
    ใส่ section label ลงใน TextBlock.extra["section"]
    ถ้า use_llm=True + มี KEY → ใช้ LLM ช่วย
    ถ้า error หรือไม่มี KEY → fallback เป็น rule-based (_guess_section_rule)
    """
    client = _get_llm_client() if use_llm else None

    if client:
        # ทำทีละก้อนใหญ่ ให้โมเดลช่วย tag section เฉพาะบาง block แรก
        joined = []
        for i, b in enumerate(doc.texts):
            joined.append(f"[{i}] {b.content}")
        prompt_text = "\n".join(joined[:200])  # limit 200 blocks แรก

        prompt = f"""
You are a document segmenter.

For each numbered text block below, assign ONE section label from:
{SECTION_LABELS}

- header        : document title / main headings
- summary       : executive summary / overview / high-level summary
- transactions  : detailed transactional / itemized content
- footer        : signatures, closing statements
- qna           : question/answer sections (e.g. "ถาม:", "ตอบ:", exam-style)
- other         : anything else

Format: one line per block, in the form:
index: label

Text blocks:
{prompt_text}
"""

        try:
            # [CHANGE] ใช้ Chat Completion API
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
            
            mapping: Dict[int, str] = {}
            for line in resp_text.splitlines():
                line = line.strip()
                if not line or ":" not in line:
                    continue
                idx_str, label = line.split(":", 1)
                idx_str = idx_str.strip().strip("[]")
                label = label.strip().lower()
                try:
                    idx = int(idx_str)
                except ValueError:
                    continue
                if label not in SECTION_LABELS:
                    label = "other"
                mapping[idx] = label

            # apply mapping + fallback rule-based ที่ไม่มีใน mapping
            total = len(doc.texts)
            for i, b in enumerate(doc.texts):
                extra = dict(b.extra or {})
                extra["section"] = mapping.get(i, _guess_section_rule(b, i, total))
                b.extra = extra

            return doc

        except Exception as e:
            print("[semantic_enricher] LLM section tagging failed:", e)
            print("[semantic_enricher] Fallback to rule-based tagging")

    # fallback: rule-based ทั้งหมด
    total = len(doc.texts)
    for i, b in enumerate(doc.texts):
        extra = dict(b.extra or {})
        extra["section"] = _guess_section_rule(b, i, total)
        b.extra = extra

    return doc


# ===========================
# 2) TEXT ROLE CATEGORIZATION
# ===========================

TEXT_ROLE_LABELS = [
    "title",                # ชื่อรายงาน / header ใหญ่
    "account_info",         # ชื่อบัญชี / เลขบัญชี / ธนาคาร
    "transaction_header",   # header ส่วนหัวของตารางรายการเดินบัญชี
    "transaction_row",      # ข้อความบรรยายรายการ (เช่น “โอนจาก XXX”)
    "note",                 # หมายเหตุ / ข้อมูลเพิ่มเติม
    "footer_text",          # ข้อความปิดท้าย
    "qna_question",         # คำถาม (ถาม:, ข้อที่..., question)
    "qna_answer",           # คำตอบ/เฉลย (ตอบ:, เฉลย)
    "other",
]


def _guess_text_role_rule(block: TextBlock) -> str:
    txt = (block.content or "").strip()
    lower = txt.lower()
    extra = block.extra or {}
    section = extra.get("section")

    is_heading = extra.get("is_heading", False)

    # Q&A roles
    t_no_space = txt.replace(" ", "")
    if t_no_space.startswith("ถาม:") or "คำถาม" in txt or "question" in lower:
        return "qna_question"
    if t_no_space.startswith("ตอบ:") or "เฉลย" in txt or "answer" in lower:
        return "qna_answer"

    # title: หัวเรื่องใหญ่
    if section == "header" and (is_heading or len(txt) < 120):
        return "title"
    if len(txt) < 80 and any(k in lower for k in ["statement", "รายงาน", "account statement"]):
        return "title"

    # account info
    if any(k in lower for k in ["เลขที่บัญชี", "account no", "account number", "branch", "ธนาคาร", "bank"]):
        return "account_info"

    # transaction header
    if any(k in lower for k in ["วันที่", "วันเดือนปี", "transaction", "ยอดคงเหลือ", "จำนวนเงิน", "amount", "credit", "debit"]):
        return "transaction_header"

    # note
    if any(k in lower for k in ["หมายเหตุ", "note:", "หมาย เหตุ", "remark"]):
        return "note"

    # footer
    if any(k in lower for k in ["ลงชื่อ", "ผู้มีอำนาจลงนาม", "ขอแสดงความนับถือ"]):
        return "footer_text"

    # transaction_row heuristic
    if section == "transactions" and 10 <= len(txt) <= 200:
        return "transaction_row"

    # qna section butไม่ได้ match patternชัดเจน
    if section == "qna" and len(txt) > 5:
        # ถ้ามี ? หรือ ตัวเลขนำหน้า + จุด → น่าจะเป็นคำถาม
        if "?" in txt or re.match(r"^\s*\d+[\).]", txt):
            return "qna_question"

    return "other"


def categorize_text_blocks(
    doc: IngestedDocument,
    use_llm: bool = False,
) -> IngestedDocument:
    """
    ใส่ role ให้ TextBlock.extra["role"] เช่น:
    - title
    - account_info
    - transaction_header
    - transaction_row
    - note
    - footer_text
    - qna_question
    - qna_answer
    - other
    """
    client = _get_llm_client() if use_llm else None

    if client:
        # ส่งเฉพาะ subset ไปให้โมเดลช่วย classify
        joined = []
        for i, b in enumerate(doc.texts[:200]):
            section = (b.extra or {}).get("section", "unknown")
            joined.append(f"[{i}] (section={section}) {b.content}")
        prompt_text = "\n".join(joined)

        prompt = f"""
You are a document text role classifier for multi-type PDFs
(e.g. bank statements, financial reports, exam Q&A, manuals).

For each text block, assign ONE role from:
{TEXT_ROLE_LABELS}

- title           : main document titles / big headings
- account_info    : bank/account information (account number, bank name, branch)
- transaction_header : header row describing transaction columns
- transaction_row : a single transaction description / row-like text
- note            : footnotes, additional explanations
- footer_text     : closing statements, signature text
- qna_question    : question text (e.g. "ถาม:", exam questions, "ข้อที่ 1...")
- qna_answer      : answer/solution text (e.g. "ตอบ:", "เฉลย")
- other           : anything else

Format: one line per block:
index: role

Text blocks:
{prompt_text}
"""

        try:
            # [CHANGE] ใช้ Chat Completion API
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
            
            mapping: Dict[int, str] = {}
            for line in resp_text.splitlines():
                line = line.strip()
                if not line or ":" not in line:
                    continue
                idx_str, label = line.split(":", 1)
                idx_str = idx_str.strip().strip("[]")
                label = label.strip().lower()
                try:
                    idx = int(idx_str)
                except ValueError:
                    continue
                if label not in TEXT_ROLE_LABELS:
                    label = "other"
                mapping[idx] = label

            for i, b in enumerate(doc.texts):
                extra = dict(b.extra or {})
                extra["role"] = mapping.get(i, _guess_text_role_rule(b))
                b.extra = extra

            return doc

        except Exception as e:
            print("[semantic_enricher] LLM text role tagging failed:", e)
            print("[semantic_enricher] Fallback to rule-based text role")

    # fallback rule-based
    for b in doc.texts:
        extra = dict(b.extra or {})
        extra["role"] = _guess_text_role_rule(b)
        b.extra = extra

    return doc


# ===========================
# 4) TABLE NORMALIZER + ROLE
# ===========================

HEADER_NORMALIZATION_MAP = {
    # date
    "date": "date",
    "วันที่": "date",
    "วันเดือนปี": "date",
    "วันที่ทำรายการ": "date",
    # description
    "description": "description",
    "details": "description",
    "รายละเอียด": "description",
    "รายการ": "description",
    "description/รายละเอียด": "description",
    # debit / credit
    "debit": "amount_out",
    "withdrawal": "amount_out",
    "withdraw": "amount_out",
    "ถอน": "amount_out",
    "จ่าย": "amount_out",
    "paid": "amount_out",
    "credit": "amount_in",
    "deposit": "amount_in",
    "ฝาก": "amount_in",
    "รับ": "amount_in",
    # balance
    "balance": "balance",
    "ยอดคงเหลือ": "balance",
    "คงเหลือ": "balance",
    "ยอดยกไป": "balance",
    # amount generic
    "amount": "amount",
    "ยอดเงิน": "amount",
    "จำนวนเงิน": "amount",
    "ยอด": "amount",
}


def _normalize_header_name(h: str) -> str:
    """normalize header ชื่อ → canonical name ถ้าเจอ"""
    h_clean = (h or "").strip().lower()
    if not h_clean:
        return ""
    for key, canonical in HEADER_NORMALIZATION_MAP.items():
        if key in h_clean:
            return canonical
    return h_clean


TABLE_ROLE_LABELS = ["transaction_table", "summary_table", "other_table"]


def _guess_table_role(tb: TableBlock) -> str:
    header = getattr(tb, "header", []) or []
    header_lower = [str(h).lower() for h in header]
    header_joined = " ".join(header_lower)

    has_date = any("date" in h or "วันที่" in h for h in header_lower)
    has_amount = any(
        any(x in h for x in ["amount", "ยอดเงิน", "debit", "credit", "ยอดคงเหลือ", "balance"])
        for h in header_lower
    )

    if has_date and has_amount:
        return "transaction_table"

    if any(k in header_joined for k in ["summary", "สรุป", "total", "รวม", "สรุปยอด"]):
        return "summary_table"

    return "other_table"


def normalize_tables(tables: List[TableBlock]) -> List[TableBlock]:
    """
    ปรับ header ของตารางให้เป็นชื่อมาตรฐาน เช่น
    - date
    - description
    - amount_in / amount_out
    - balance

    และใส่ role ลงใน TableBlock.extra["role"]
    """
    for tb in tables:
        header = list(getattr(tb, "header", []) or [])
        rows = list(getattr(tb, "rows", []) or [])

        # ถ้า header ว่าง แต่แถวแรกดูเหมือนเป็น header (ไม่ใช่ตัวเลขล้วน ๆ)
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
        extra_norm.update(
            {
                "original_header": header,
                "normalized_header": normalized_header,
            }
        )
        extra["header_normalization"] = extra_norm

        # ใส่ role ให้ table ด้วย
        extra["role"] = _guess_table_role(tb)
        tb.extra = extra

    return tables


# ===========================
# 5) MAPPING PREPARE (transactions)
# ===========================


def _parse_float_safe(val: Optional[str]) -> Optional[float]:
    if val is None:
        return None
    s = str(val).strip()
    # ตัดสัญลักษณ์ที่ชอบติดมา
    s = s.replace(",", "").replace("฿", "")
    # รูปแบบ (123.45) = -123.45
    if s.startswith("(") and s.endswith(")"):
        s = "-" + s[1:-1]
    # กันเคสมี space แปลก ๆ
    s = s.replace(" ", "")
    try:
        return float(s)
    except ValueError:
        return None


def extract_transactions_from_table(tb: TableBlock) -> List[Dict[str, Any]]:
    """
    พยายาม map ตารางให้กลายเป็น transaction records:
    - หา column index ของ date / description / amount / amount_in / amount_out / balance
    - คืน list ของ dict ที่มี key เหล่านี้
    """
    header = getattr(tb, "header", []) or []
    rows = getattr(tb, "rows", []) or []

    # map header → index
    name_to_idx: Dict[str, int] = {}
    for i, h in enumerate(header):
        if not h:
            continue
        name_to_idx[h] = i

    records: List[Dict[str, Any]] = []

    for row in rows:
        def col(name: str) -> Optional[str]:
            idx = name_to_idx.get(name)
            if idx is None or idx >= len(row):
                return None
            return str(row[idx]).strip()

        date = col("date")
        desc = col("description")

        amount_in = col("amount_in")
        amount_out = col("amount_out")
        amount = col("amount")
        balance = col("balance")

        if not any([date, desc, amount_in, amount_out, amount, balance]):
            continue

        record: Dict[str, Any] = {
            "date_raw": date,
            "description": desc,
            "amount_in_raw": amount_in,
            "amount_out_raw": amount_out,
            "amount_raw": amount,
            "balance_raw": balance,
            "amount_in": _parse_float_safe(amount_in) if amount_in else None,
            "amount_out": _parse_float_safe(amount_out) if amount_out else None,
            "amount": _parse_float_safe(amount) if amount else None,
            "balance": _parse_float_safe(balance) if balance else None,
        }

        records.append(record)

    return records


def prepare_mapping_payload(doc: IngestedDocument) -> Dict[str, Any]:
    """
    ดึงข้อมูลที่จำเป็นสำหรับทำ mapping ข้ามเอกสาร:
    - doc metadata
    - transaction records (จากตารางที่ normalize แล้ว)
    """
    all_transactions: List[Dict[str, Any]] = []

    for tb in doc.tables:
        txs = extract_transactions_from_table(tb)
        if not txs:
            continue
        all_transactions.extend(txs)

    payload: Dict[str, Any] = {
        "doc_id": doc.metadata.doc_id,
        "doc_type": doc.metadata.doc_type,
        "file_name": doc.metadata.file_name,
        "transactions": all_transactions,
    }
    return payload