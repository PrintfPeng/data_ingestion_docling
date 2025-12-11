from __future__ import annotations
from typing import List, Optional
import os
from ingestion.schema import IngestedDocument, TextBlock

CANDIDATE_TYPES = [
    "bank_statement", "invoice", "receipt", "purchase_order",
    "delivery_note", "tax_form", "qna", "generic",
]

# ใช้โมเดลนี้เป็นหลัก
PRIMARY_MODEL = "models/gemini-2.5-flash"
MODEL_CANDIDATES = [PRIMARY_MODEL]

def _collect_sample_text(texts: List[TextBlock], max_chars: int = 4000) -> str:
    chunks = []
    total = 0
    for t in texts:
        if not t.content:
            continue
        if total + len(t.content) > max_chars:
            break
        chunks.append(t.content)
        total += len(t.content)
    return "\n".join(chunks)

def _get_gemini_api_key() -> Optional[str]:
    api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    return api_key

def classify_document_rule_based(doc: IngestedDocument) -> str:
    file_name = (doc.metadata.file_name or "").lower()
    sample = _collect_sample_text(doc.texts).lower()

    if any(k in file_name for k in ["qna", "q&a", "qa", "quiz", "exam", "ข้อสอบ", "แบบฝึกหัด"]):
        return "qna"
    if ("ถาม:" in sample and "ตอบ:" in sample) or ("คำถาม" in sample and "คำตอบ" in sample):
        return "qna"
    if "statement" in file_name and any(k in file_name for k in ["bank", "acct", "account", "บัญชี"]):
        return "bank_statement"
    if "invoice" in file_name: return "invoice"
    if "receipt" in file_name: return "receipt"
    if "po_" in file_name or "purchase_order" in file_name: return "purchase_order"
    if "delivery" in file_name or "dnote" in file_name: return "delivery_note"
    
    if any(k in sample for k in ["account statement", "statement period", "เลขที่บัญชี", "รายการเดินบัญชี"]):
        return "bank_statement"
    if any(k in sample for k in ["invoice no", "tax invoice", "ใบกำกับภาษี"]):
        return "invoice"
    if any(k in sample for k in ["receipt no", "official receipt", "ใบเสร็จรับเงิน"]):
        return "receipt"
    if any(k in sample for k in ["purchase order", "ใบสั่งซื้อ"]):
        return "purchase_order"
    if any(k in sample for k in ["delivery note", "ใบส่งของ"]):
        return "delivery_note"
    if any(k in sample for k in ["tax form", "withholding tax", "หนังสือรับรองการหักภาษี"]):
        return "tax_form"

    return "generic"

def classify_document_with_gemini(doc: IngestedDocument, model_name: Optional[str] = None) -> str:
    try:
        import google.generativeai as genai
    except Exception:
        return classify_document_rule_based(doc)

    api_key = _get_gemini_api_key()
    if not api_key:
        return classify_document_rule_based(doc)

    try:
        genai.configure(api_key=api_key)
    except Exception:
        return classify_document_rule_based(doc)

    if model_name is None:
        model_name = PRIMARY_MODEL

    sample_text = _collect_sample_text(doc.texts, max_chars=4000)
    prompt = f"""
    คุณเป็นตัวช่วยจำแนกประเภทเอกสาร (PDF) ภาษาไทยและอังกฤษ
    จำแนกเอกสารด้านล่างนี้เป็น "ประเภทเดียว" จากลิสต์นี้: {CANDIDATE_TYPES}
    File name: {doc.metadata.file_name}
    Text: \"\"\"{sample_text}\"\"\"
    ตอบแค่ชื่อ label ภาษาอังกฤษตัวพิมพ์เล็กจากลิสต์ข้างบน เช่น 'bank_statement' หรือ 'generic'
    """

    try:
        model = genai.GenerativeModel(model_name)
        resp = model.generate_content(prompt)
        answer = (getattr(resp, "text", "") or "").strip().lower()
        
        answer = answer.replace("label:", "").strip()
        for lbl in CANDIDATE_TYPES:
            if lbl in answer:
                return lbl
        return "generic"
    except Exception as e:
        print(f"[document_classifier] Gemini classify failed: {e}")
        return classify_document_rule_based(doc)

def classify_document(doc: IngestedDocument, use_gemini: bool = True) -> str:
    if not doc.texts:
        return classify_document_rule_based(doc)
    if not use_gemini:
        return classify_document_rule_based(doc)
    return classify_document_with_gemini(doc)