from __future__ import annotations
from typing import List, Dict, Any, Optional
import os
import re
from .schema import IngestedDocument, TextBlock, TableBlock

# ใช้รุ่นเดียวกันทั้งโปรเจกต์
GEMINI_MODEL = "models/gemini-2.5-flash"

def _get_gemini_model():
    api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    if not api_key:
        return None
    try:
        import google.generativeai as genai
        genai.configure(api_key=api_key)
        return genai.GenerativeModel(GEMINI_MODEL)
    except Exception as e:
        print("[semantic_enricher] Cannot init Gemini:", e)
        return None

# ===========================
# 1) SECTION TAGGING
# ===========================
SECTION_LABELS = ["header", "summary", "transactions", "footer", "qna", "other"]
_QNA_HINTS = ["ถาม:", "คำถาม", "ข้อที่", "จงตอบ", "เลือกคำตอบ", "question"]

def _looks_like_qna(text: str) -> bool:
    t = text.replace(" ", "")
    return any(h in t for h in _QNA_HINTS)

def _guess_section_rule(block: TextBlock, index: int, total: int) -> str:
    txt = (block.content or "").strip()
    lower = txt.lower()
    extra = block.extra or {}
    is_heading = bool(extra.get("is_heading"))
    page = getattr(block, "page", None) or 0

    if _looks_like_qna(txt): return "qna"
    if is_heading and (index < 10 or page <= 2): return "header"
    if index == 0 and len(txt) <= 120: return "header"
    if any(k in lower for k in ["summary", "สรุป", "overview"]): return "summary"
    if any(k in lower for k in ["รายการ", "statement", "transactions"]): return "transactions"
    if any(k in lower for k in ["ลงชื่อ", "signature", "ขอแสดงความนับถือ"]): return "footer"
    return "other"

def tag_sections(doc: IngestedDocument, use_gemini: bool = False) -> IngestedDocument:
    model = _get_gemini_model() if use_gemini else None
    if model:
        joined = []
        for i, b in enumerate(doc.texts):
            joined.append(f"[{i}] {b.content}")
        prompt_text = "\n".join(joined[:200])
        prompt = f"""
        You are a document segmenter. Assign ONE label from {SECTION_LABELS} for each block.
        Format: index: label
        Text blocks: {prompt_text}
        """
        try:
            resp = model.generate_content(prompt)
            mapping = {}
            for line in (resp.text or "").splitlines():
                if ":" in line:
                    idx_str, label = line.split(":", 1)
                    try:
                        mapping[int(idx_str.strip().strip("[]"))] = label.strip().lower()
                    except: continue
            
            total = len(doc.texts)
            for i, b in enumerate(doc.texts):
                extra = dict(b.extra or {})
                extra["section"] = mapping.get(i, _guess_section_rule(b, i, total))
                b.extra = extra
            return doc
        except Exception as e:
            print("[semantic_enricher] Gemini section tagging failed:", e)

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
    "title", "account_info", "transaction_header", "transaction_row",
    "note", "footer_text", "qna_question", "qna_answer", "other"
]

def _guess_text_role_rule(block: TextBlock) -> str:
    txt = (block.content or "").strip()
    lower = txt.lower()
    extra = block.extra or {}
    section = extra.get("section")
    t_no_space = txt.replace(" ", "")
    
    if t_no_space.startswith("ถาม:") or "คำถาม" in txt: return "qna_question"
    if t_no_space.startswith("ตอบ:") or "เฉลย" in txt: return "qna_answer"
    if section == "header" and len(txt) < 120: return "title"
    if any(k in lower for k in ["เลขที่บัญชี", "account no", "ธนาคาร"]): return "account_info"
    if any(k in lower for k in ["วันที่", "date", "balance", "ยอดคงเหลือ"]): return "transaction_header"
    if any(k in lower for k in ["หมายเหตุ", "note"]): return "note"
    if section == "footer": return "footer_text"
    return "other"

def categorize_text_blocks(doc: IngestedDocument, use_gemini: bool = False) -> IngestedDocument:
    model = _get_gemini_model() if use_gemini else None
    if model:
        # (Similar logic to tag_sections but for roles) - Simplified for brevity
        pass 

    for b in doc.texts:
        extra = dict(b.extra or {})
        extra["role"] = _guess_text_role_rule(b)
        b.extra = extra
    return doc

# ===========================
# 4) TABLE NORMALIZER
# ===========================
HEADER_NORMALIZATION_MAP = {
    "date": "date", "วันที่": "date", "วันเดือนปี": "date",
    "description": "description", "รายการ": "description",
    "debit": "amount_out", "ถอน": "amount_out",
    "credit": "amount_in", "ฝาก": "amount_in",
    "balance": "balance", "คงเหลือ": "balance", "amount": "amount"
}

def _normalize_header_name(h: str) -> str:
    h_clean = (h or "").strip().lower()
    for key, canonical in HEADER_NORMALIZATION_MAP.items():
        if key in h_clean: return canonical
    return h_clean

def normalize_tables(tables: List[TableBlock]) -> List[TableBlock]:
    for tb in tables:
        header = list(getattr(tb, "header", []) or [])
        tb.header = [_normalize_header_name(h) for h in header]
        extra = dict(tb.extra or {})
        extra["role"] = "transaction_table" if "date" in tb.header else "other_table"
        tb.extra = extra
    return tables

# ===========================
# 5) MAPPING
# ===========================
def prepare_mapping_payload(doc: IngestedDocument) -> Dict[str, Any]:
    return {
        "doc_id": doc.metadata.doc_id,
        "doc_type": doc.metadata.doc_type,
        "file_name": doc.metadata.file_name,
        "transactions": []
    }