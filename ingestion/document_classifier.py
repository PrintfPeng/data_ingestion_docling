from __future__ import annotations

"""
document_classifier.py (Final Universal Edition)

หน้าที่:
- จำแนกประเภทเอกสารจากข้อความและชื่อไฟล์ (ครอบคลุมหลากหลายประเภท)
- รองรับ 3 โหมดทำงาน (Hybrid Fallback System):
    1) OpenRouter (AI ฉลาดสุด) -> ถ้าพังไปข้อ 2
    2) Google Gemini (AI สำรอง ฟรี/เร็ว) -> ถ้าพังไปข้อ 3
    3) Rule-based (Keyword Matching) -> กันตาย

Updated Categories:
- Finance: invoice, receipt, financial_statement, tax_form
- Legal/Admin: contract, government_doc, id_card
- Work/HR: resume, meeting_minutes, project_plan
- Knowledge: manual, research_paper, educational
- General: correspondence, generic
"""

from typing import List, Optional
from dotenv import load_dotenv
import os
import re

load_dotenv()

from ingestion.schema import IngestedDocument, TextBlock, DocumentMetadata

# -------------------------
# Client Imports (Safe Load)
# -------------------------
try:
    from openai import OpenAI
except ImportError:
    OpenAI = None

try:
    import google.generativeai as genai
    HAS_GOOGLE = True
except ImportError:
    HAS_GOOGLE = False

# -------------------------
# Document Label Set (Universal)
# -------------------------
CANDIDATE_TYPES = [
    # Finance
    "invoice",              # ใบแจ้งหนี้
    "receipt",              # ใบเสร็จรับเงิน
    "financial_statement",  # งบการเงิน / Bank Statement
    "tax_form",             # ภาษี
    
    # Legal & Official
    "contract",             # สัญญา / ข้อตกลง
    "government_doc",       # หนังสือราชการ / ประกาศ
    "id_card",              # บัตรประชาชน / Passport
    
    # Work & HR
    "resume",               # เรซูเม่ / CV
    "meeting_minutes",      # บันทึกการประชุม
    "project_plan",         # แผนงาน / TOR
    
    # Knowledge & Tech
    "manual",               # คู่มือ / Technical Spec
    "research_paper",       # งานวิจัย / บทความวิชาการ
    "educational",          # สื่อการสอน / ข้อสอบ / แบบฝึกหัด
    
    # General
    "correspondence",       # จดหมาย / อีเมล / บันทึกข้อความ
    "generic",              # ทั่วไป
]

# -------------------------
# Model Config
# -------------------------
PRIMARY_MODEL = os.getenv("CUSTOM_MODEL_NAME", "qwen/qwen-2.5-72b-instruct")

# -------------------------
# HELPER: Client Managers
# -------------------------

def _get_openai_client() -> Optional[OpenAI]:
    """สร้าง Client สำหรับ OpenRouter"""
    api_key = os.getenv("CUSTOM_API_KEY")
    base_url = os.getenv("CUSTOM_API_BASE")
    if not api_key: return None
    # Check library import
    if OpenAI is None: return None
    try:
        return OpenAI(api_key=api_key, base_url=base_url, timeout=30)
    except: return None

def _get_google_client():
    """สร้าง Client สำหรับ Google Gemini"""
    if not HAS_GOOGLE: return None
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key: return None
    try:
        genai.configure(api_key=api_key)
        # ใช้โมเดล Flash เพราะฟรีและเร็ว
        return genai.GenerativeModel('gemini-2.5-flash')
    except: return None

# -------------------------
# HELPER: Text Sampling
# -------------------------

def _collect_sample_text(texts: List[TextBlock], max_chars: int = 4000) -> str:
    """รวม text block แรก ๆ เอามาเป็น sample text"""
    chunks = []
    current_len = 0
    # ดูแค่ 30 บล็อกแรก (เพิ่มจากเดิมเผื่อเอกสารยาว)
    for t in texts[:30]: 
        s = (t.content or "").strip()
        if not s: continue
        chunks.append(s)
        current_len += len(s)
        if current_len >= max_chars: break
    return "\n".join(chunks)[:max_chars]

# -------------------------
# LOGIC: Rule-based (Universal)
# -------------------------

def classify_document_rule_based(doc: IngestedDocument) -> str:
    """
    ใช้ Keyword พื้นฐานแยกประเภท (Fallback ชั้นสุดท้าย)
    รองรับทั้งไทยและอังกฤษ
    """
    text = _collect_sample_text(doc.texts).lower()
    fname = (doc.metadata.file_name or "").lower()
    combined = f"{fname} {text}"

    # 1. Finance
    if any(k in combined for k in ["invoice", "ใบแจ้งหนี้", "tax invoice"]): return "invoice"
    if any(k in combined for k in ["receipt", "ใบเสร็จ", "bill"]): return "receipt"
    if any(k in combined for k in ["statement", "รายการเดินบัญชี", "งบดุล", "balance sheet"]): return "financial_statement"
    if any(k in combined for k in ["tax", "ภาษี", "ภ.ง.ด", "withholding"]): return "tax_form"

    # 2. Legal
    if any(k in combined for k in ["contract", "agreement", "สัญญา", "ข้อตกลง", "mou"]): return "contract"
    if any(k in combined for k in ["identification", "passport", "บัตรประชาชน", "citizen id"]): return "id_card"
    if any(k in combined for k in ["official", "ประกาศ", "ระเบียบ", "คำสั่ง", "gazette"]): return "government_doc"

    # 3. Work
    if any(k in combined for k in ["resume", "cv", "curriculum vitae", "ประวัติย่อ", "experience"]): return "resume"
    if any(k in combined for k in ["minutes", "บันทึกการประชุม", "agenda", "วาระ"]): return "meeting_minutes"
    
    # 4. Knowledge
    if any(k in combined for k in ["manual", "guide", "handbook", "คู่มือ", "instruction", "spec"]): return "manual"
    if any(k in combined for k in ["abstract", "introduction", "methodology", "บทคัดย่อ", "วิจัย"]): return "research_paper"
    if any(k in combined for k in ["exam", "test", "quiz", "ข้อสอบ", "แบบฝึกหัด", "lesson"]): return "educational"

    return "generic"

# -------------------------
# LOGIC: Hybrid LLM Classification
# -------------------------

def classify_document_with_llm(doc: IngestedDocument) -> str:
    """
    Hybrid Classification: OpenRouter -> Google -> Rule-based
    """
    sample_text = _collect_sample_text(doc.texts)
    # ถ้าไม่มี Text เลย ให้ใช้ Rule-based (ซึ่งจะดูชื่อไฟล์แทน)
    if not sample_text: return classify_document_rule_based(doc)

    file_name = doc.metadata.file_name or ""
    
    prompt = (
        f"Analyze this document content and filename.\n"
        f"Filename: {file_name}\n"
        f"Content Sample (First 4000 chars):\n{sample_text}\n\n"
        f"Classify into exactly one of these types:\n"
        f"{', '.join(CANDIDATE_TYPES)}\n\n"
        f"Guidelines:\n"
        f"- 'contract': Legal agreements, MOUs\n"
        f"- 'manual': User guides, technical specs, handbooks\n"
        f"- 'research_paper': Academic papers, journals\n"
        f"- 'correspondence': Letters, emails, memos\n"
        f"- 'generic': If unsure or general text\n"
        f"\nReply ONLY with the type name (lowercase snake_case)."
    )

    # 1. แผน A: OpenRouter (Primary)
    client = _get_openai_client()
    if client:
        try:
            res = client.chat.completions.create(
                model=PRIMARY_MODEL,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=50, temperature=0.0
            )
            t = res.choices[0].message.content.strip().lower()
            t = re.sub(r"[^a-z_]", "", t)
            if t in CANDIDATE_TYPES: return t
        except Exception as e:
            print(f"[classifier] OpenRouter failed: {e}")

    # 2. แผน B: Google Gemini (Fallback)
    google = _get_google_client()
    if google:
        try:
            print("[classifier] 🔄 Using Google Fallback...")
            res = google.generate_content(prompt)
            t = res.text.strip().lower()
            t = re.sub(r"[^a-z_]", "", t)
            if t in CANDIDATE_TYPES: return t
        except Exception as e:
            print(f"[classifier] Google failed: {e}")

    # 3. แผน C: Rule-based (กันตาย)
    print("[classifier] AI failed, falling back to rules.")
    return classify_document_rule_based(doc)

# -------------------------
# PUBLIC ENTRYPOINT
# -------------------------

def classify_document(doc: IngestedDocument, use_llm: bool = True) -> str:
    """
    Entrypoint หลักสำหรับเรียกใช้งานจากภายนอก
    """
    # ถ้าเอกสารว่างเปล่า (ไม่มี text) ให้ใช้ rule ดูชื่อไฟล์
    if not doc.texts:
        return classify_document_rule_based(doc)

    # ถ้าสั่งปิด LLM ให้ใช้ rule
    if not use_llm:
        return classify_document_rule_based(doc)

    # ปกติใช้ Hybrid LLM
    return classify_document_with_llm(doc)

# -------------------------
# CLI TEST (สำหรับรันเทสไฟล์นี้เดี่ยวๆ)
# -------------------------

if __name__ == "__main__":
    import json
    from pathlib import Path

    # path สมมติสำหรับการเทส
    root = Path("ingested") / "sample"
    meta_path = root / "metadata.json"
    text_path = root / "text.json"

    if not meta_path.exists() or not text_path.exists():
        print("Test files not found. Please run ingestion first.")
    else:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        texts = json.loads(text_path.read_text(encoding="utf-8"))

        doc = IngestedDocument(
            metadata=DocumentMetadata.from_dict(meta),
            texts=[TextBlock.from_dict(t) for t in texts],
            tables=[],
            images=[],
        )

        print("-" * 50)
        print(f"File: {doc.metadata.filename}")
        print("-" * 50)
        print("Rule-based Result:", classify_document_rule_based(doc))
        print("AI Result (Hybrid):", classify_document(doc, use_llm=True))