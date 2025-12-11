from __future__ import annotations

"""
document_classifier.py

หน้าที่:
- จำแนกประเภทเอกสารจากข้อความ text blocks และชื่อไฟล์
- รองรับ 2 โหมด:
    1) Rule-based (ไม่ใช้โมเดล)
    2) Gemini LLM-based (ใช้โมเดล gemini-2.5 / 2.0)

ขั้นตอน:
- อ่าน TextBlock
- รวมข้อความบางส่วน (sample_text)
- Rule-based → ถ้าดูไม่ออก
- ถ้า use_gemini=True → ใช้ LLM ช่วย classify
"""

from typing import List, Optional
import os

from ingestion.schema import IngestedDocument, TextBlock, DocumentMetadata

# -------------------------
# Document Label Set
# -------------------------
CANDIDATE_TYPES = [
    "bank_statement",
    "invoice",
    "receipt",
    "purchase_order",
    "delivery_note",
    "tax_form",
    "qna",          # เพิ่ม type สำหรับเอกสารแนวถาม-ตอบ / แบบฝึกหัด
    "generic",
]

# -------------------------
# Gemini Model Candidates
# -------------------------
# เราจะพยายามใช้ PRO ก่อน ถ้าไม่ได้ค่อย fallback เป็น flash
PRIMARY_MODEL = "models/gemini-2.5-pro"
MODEL_CANDIDATES = [
    PRIMARY_MODEL,
    "models/gemini-2.5-flash",
]

# -------------------------
# HELPER FUNCTION
# -------------------------


def _collect_sample_text(texts: List[TextBlock], max_chars: int = 4000) -> str:
    """รวม text block แรก ๆ เอามาเป็น sample text สำหรับ rule/LLM"""
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
    """
    ดึง API KEY แบบยืดหยุ่น:
    - ลอง GEMINI_API_KEY ก่อน
    - ถ้าไม่มี ใช้ GOOGLE_API_KEY แทน (ส่วนใหญ่จะเป็น key เดียวกัน)
    """
    api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    prefix = (api_key or "None")[:10]
    print(f"[document_classifier] GEMINI/GOOGLE key prefix: {prefix}")
    return api_key


# ============================================================
# 1) RULE-BASED CLASSIFIER (พื้นฐาน)
# ============================================================


def classify_document_rule_based(doc: IngestedDocument) -> str:
    """จำแนกเอกสารแบบง่าย ๆ ไม่ใช้ AI"""
    file_name = (doc.metadata.file_name or "").lower()
    sample = _collect_sample_text(doc.texts).lower()

    # ------------------------
    # 1) Q&A / แบบฝึกหัด / ข้อสอบ
    # ------------------------
    # ใช้ทั้งจากชื่อไฟล์ + เนื้อหา
    if any(k in file_name for k in ["qna", "q&a", "qa", "quiz", "exam", "ข้อสอบ", "แบบฝึกหัด"]):
        return "qna"

    if ("ถาม:" in sample and "ตอบ:" in sample) or ("คำถาม" in sample and "คำตอบ" in sample):
        return "qna"

    # ------------------------
    # 2) rule จากชื่อไฟล์
    # ------------------------
    if "statement" in file_name and "bank" in file_name:
        return "bank_statement"

    if "statement" in file_name and any(k in file_name for k in ["acct", "account", "บัญชี"]):
        return "bank_statement"

    if "invoice" in file_name:
        return "invoice"

    if "receipt" in file_name:
        return "receipt"

    if "po_" in file_name or "purchase_order" in file_name:
        return "purchase_order"

    if "delivery" in file_name or "dnote" in file_name:
        return "delivery_note"

    # ------------------------
    # 3) rule จากเนื้อหา (ภาษาอังกฤษ + ไทย)
    # ------------------------
    # bank statement
    if any(k in sample for k in [
        "account statement",
        "statement period",
        "account number",
        "เลขที่บัญชี",
        "ยอดคงเหลือ",
        "รายการเดินบัญชี",
        "รายการเคลื่อนไหวบัญชี",
    ]):
        return "bank_statement"

    # invoice
    if any(k in sample for k in [
        "invoice no",
        "tax invoice",
        "เลขที่ใบกำกับภาษี",
        "เลขที่ใบแจ้งหนี้",
    ]):
        return "invoice"

    # receipt
    if any(k in sample for k in [
        "receipt no",
        "official receipt",
        "thank you for your payment",
        "ใบเสร็จรับเงิน",
    ]):
        return "receipt"

    # purchase order
    if any(k in sample for k in [
        "purchase order",
        "ใบสั่งซื้อ",
    ]):
        return "purchase_order"

    # delivery note
    if any(k in sample for k in [
        "delivery note",
        "ใบส่งของ",
        "ใบส่งสินค้า",
    ]):
        return "delivery_note"

    # tax form
    if any(k in sample for k in [
        "tax form",
        "withholding tax",
        "หนังสือรับรองการหักภาษี ณ ที่จ่าย",
    ]):
        return "tax_form"

    # Q&A อีกที (สำรอง)
    if "ถาม:" in sample and "ตอบ:" in sample:
        return "qna"

    return "generic"


# ============================================================
# 2) GEMINI-BASED CLASSIFIER
# ============================================================


def classify_document_with_gemini(
    doc: IngestedDocument,
    model_name: Optional[str] = None,
) -> str:
    """
    ใช้ Gemini จำแนกประเภทเอกสาร
    - ใช้โมเดล fix (PRIMARY_MODEL) ถ้าไม่กำหนด
    - ถ้า error / ไม่มี KEY → fallback rule-based
    """
    try:
        import google.generativeai as genai
    except Exception as e:
        print(f"[document_classifier] google.generativeai import failed: {e}")
        return classify_document_rule_based(doc)

    api_key = _get_gemini_api_key()
    if not api_key:
        print("[document_classifier] No API KEY → fallback rule-based")
        return classify_document_rule_based(doc)

    try:
        genai.configure(api_key=api_key)
    except Exception as e:
        print(f"[document_classifier] genai.configure failed: {e}")
        return classify_document_rule_based(doc)

    # เลือก model
    if model_name is None:
        model_name = PRIMARY_MODEL

    # ถ้าส่งชื่อมาแปลก ๆ ให้ยังมี candidate list ช่วยสำรอง
    candidates = [model_name] + [m for m in MODEL_CANDIDATES if m != model_name]

    # เตรียมข้อความ
    sample_text = _collect_sample_text(doc.texts, max_chars=4000)

    prompt = f"""
คุณเป็นตัวช่วยจำแนกประเภทไฟล์เอกสาร (PDF) ภาษาไทยและอังกฤษ

ให้จำแนกเอกสารด้านล่างนี้เป็น "ประเภทเดียว" จากลิสต์นี้เท่านั้น (ตอบเป็นภาษาอังกฤษ, ใช้ label ด้านล่างตรง ๆ):

{CANDIDATE_TYPES}

คำอธิบายแบบย่อ:
- bank_statement  = รายการเดินบัญชีธนาคาร / statement ธนาคาร
- invoice         = ใบแจ้งหนี้ / ใบกำกับภาษีขาย
- receipt         = ใบเสร็จรับเงิน
- purchase_order  = ใบสั่งซื้อ
- delivery_note   = ใบส่งของ / ใบส่งสินค้า
- tax_form        = แบบฟอร์มภาษี / หนังสือรับรองการหักภาษี ฯลฯ
- qna             = เอกสารที่เป็นชุดคำถาม–คำตอบ, ข้อสอบ, แบบฝึกหัด (มักมีรูปแบบ "ถาม:" และ "ตอบ:")
- generic         = เอกสารทั่วไปที่ไม่เข้าข้อไหนชัดเจน

File name: {doc.metadata.file_name}

ตัวอย่างข้อความจากเอกสาร:
\"\"\"{sample_text}\"\"\"

ให้ตอบแค่ชื่อ label เดียวจากลิสต์ด้านบน เช่น:
bank_statement
หรือ
qna
หรือ
generic
"""

    last_error = None

    for m in candidates:
        try:
            print(f"[document_classifier] Using Gemini model: {m}")
            model = genai.GenerativeModel(m)
            resp = model.generate_content(prompt)
            answer = (getattr(resp, "text", "") or "").strip().lower()
            print("[document_classifier] Gemini raw answer:", answer)

            # normalize
            answer = answer.replace("label:", "").strip()
            answer = answer.splitlines()[0].strip() if answer else ""

            # mappingแบบหยาบกันหลุด
            if "bank" in answer and "statement" in answer:
                return "bank_statement"
            if "invoice" in answer:
                return "invoice"
            if "receipt" in answer:
                return "receipt"
            if "purchase" in answer:
                return "purchase_order"
            if "delivery" in answer:
                return "delivery_note"
            if "tax" in answer:
                return "tax_form"
            if "qna" in answer or "q&a" in answer or "qa" in answer or "question" in answer:
                return "qna"

            # ถ้าโมเดลตอบมาหนึ่งใน label อยู่แล้วก็ใช้เลย
            for lbl in CANDIDATE_TYPES:
                if lbl in answer:
                    return lbl

            # ไม่เข้าอะไรเลย → generic
            return "generic"

        except Exception as e:
            last_error = e
            print(f"[document_classifier] Gemini classify failed with model='{m}': {e}")
            # ลองตัวถัดไปใน candidates

    print("[document_classifier] All Gemini attempts failed, fallback to rule-based.")
    if last_error:
        print(f"[document_classifier] Last error: {last_error}")
    return classify_document_rule_based(doc)


# ============================================================
# PUBLIC ENTRYPOINT
# ============================================================


def classify_document(doc: IngestedDocument, use_gemini: bool = True) -> str:
    """
    เลือกว่าจะใช้ rule-based หรือ Gemini
    """
    # กันกรณีไม่มี text เลย ยังให้ได้ type กลับไป (มักจะ generic)
    if not doc.texts:
        return classify_document_rule_based(doc)

    if not use_gemini:
        return classify_document_rule_based(doc)

    # พยายามใช้ Gemini ก่อน
    return classify_document_with_gemini(doc)


# ============================================================
# CLI TEST
# ============================================================

if __name__ == "__main__":
    import json
    from pathlib import Path

    # ทดสอบโหลดจาก ingested/sample (ต้องมีไฟล์ก่อน)
    root = Path("ingested") / "sample"
    meta_path = root / "metadata.json"
    text_path = root / "text.json"

    if not meta_path.exists() or not text_path.exists():
        print("Please run ingestion first: ingested/sample/metadata.json + text.json not found.")
    else:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        texts = json.loads(text_path.read_text(encoding="utf-8"))

        doc = IngestedDocument(
            metadata=DocumentMetadata.from_dict(meta),
            texts=[TextBlock.from_dict(t) for t in texts],
            tables=[],
            images=[],
        )

        print("Rule-based:", classify_document(doc, use_gemini=False))
        print("Gemini:", classify_document(doc, use_gemini=True))
