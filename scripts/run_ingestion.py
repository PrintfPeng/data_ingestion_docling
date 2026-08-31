# scripts/run_ingestion.py
from __future__ import annotations

import json
import argparse
from pathlib import Path
import sys
import re
from typing import Optional, List
import os

# --- [NEW] Imports สำหรับ LangChain & Gemini (OCR Corrector) ---
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate

# [CHANGE] ใช้ DoclingParser ตัวใหม่ที่เราทำ Hybrid Ingestion ไว้
from ingestion.docling_parser import DoclingParser
from ingestion.document_classifier import classify_document
from ingestion.image_extractor import extract_images
from ingestion.schema import IngestedDocument, TextBlock
from ingestion.validator import validate_all
from ingestion.ocr_extractor import ocr_extract_document
# [IMPORTANT] ต้องเรียกตัวนี้มาใช้ ไม่งั้นไฟล์ .md ไม่เกิด
from ingestion.markdown_generator import generate_markdown

# Helper Class สำหรับส่ง Config ให้ DoclingParser (เพื่อให้เซฟรูปได้ถูกที่)
class IngestionConfig:
    def __init__(self, output_dir: str):
        self.output_dir = output_dir


# =====================================================================
# [NEW] AI OCR Corrector: ฟังก์ชันใช้ LLM แก้คำผิดจากการสแกน OCR
# =====================================================================
def _correct_ocr_text_with_llm(raw_text: str) -> str:
    """
    ใช้ LLM (Gemini) ในการแก้คำผิดจากการสแกน OCR 
    โดยรักษาตัวเลขและโครงสร้างประโยคเดิมไว้
    """
    if not raw_text or len(raw_text.strip()) < 10:
        return raw_text # ถ้ายาวไม่พอ ไม่ต้องส่งแก้ให้เปลืองโควต้า

    try:
        # ใช้ Flash เพื่อความรวดเร็วและประหยัดค่าใช้จ่าย
        # หมายเหตุ: ต้องมีตัวแปร GOOGLE_API_KEY อยู่ใน Environment
        llm = ChatGoogleGenerativeAI(
            model="gemini-1.5-flash", 
            temperature=0.0, # บังคับให้ไม่คิดเองเออเอง (No Hallucination)
            max_tokens=2048
        )

        # ออกแบบ Prompt แบบ Zero-shot Instruction
        prompt = PromptTemplate.from_template(
            """คุณคือผู้เชี่ยวชาญด้านการตรวจสอบและแก้ไขเอกสารภาษาไทย (Proofreader)
ข้อความด้านล่างนี้เป็นข้อความที่ได้จากระบบ OCR ซึ่งมีคำผิด (Typo) สระลอย หรือพยัญชนะที่อ่านผิดเพี้ยนไป

หน้าที่ของคุณ:
1. แก้ไขคำที่สะกดผิดให้ถูกต้องตามบริบทภาษาไทย (เช่น "เฉินทถแทน" -> "เงินทดแทน", "จานวน" -> "จำนวน")
2. ห้ามดัดแปลงแก้ไข "ตัวเลข" "ชื่อเฉพาะ" หรือ "วันที่" โดยเด็ดขาด
3. ห้ามตัดทอนข้อความ หรือสรุปความ ให้คงโครงสร้างประโยคเดิมไว้ให้มากที่สุด
4. ส่งคืนกลับมา *เฉพาะข้อความที่แก้ไขแล้วเท่านั้น* ห้ามมีคำอธิบาย หรือเครื่องหมาย Markdown ครอบข้อความ

ข้อความต้นฉบับ:
{raw_text}
"""
        )

        chain = prompt | llm
        corrected_text = chain.invoke({"raw_text": raw_text}).content

        # ลบ Whitespace ส่วนเกินที่ LLM อาจจะเผลอใส่มา
        return corrected_text.strip()

    except Exception as e:
        print(f"⚠️ [OCR-Corrector] LLM Correction failed, using raw text. Error: {e}")
        return raw_text # Fallback: ถ้า AI พัง ให้คืนค่าข้อความดิบกลับไป ป้องกันระบบล่ม
# =====================================================================


def _attach_ocr_text(doc: IngestedDocument, pdf_path: Path, use_ocr: bool = True) -> None:
    """
    เรียก OCR ด้วย Gemini แล้วเอาข้อความต่อเข้าไปใน doc.texts
    พร้อมทั้งทำความสะอาดภาษาไทยและแก้คำผิดด้วย AI
    """
    if not use_ocr:
        return

    try:
        # หมายเหตุ: Docling มี OCR ในตัวอยู่แล้ว (Tesseract)
        # แต่ถ้าอยากใช้ Gemini OCR เสริมอีก ก็เปิดไว้ได้ครับ
        # (ฟังก์ชันนี้จะเรียก ocr_extract_document จาก ingestion/ocr_extractor.py)
        ocr_result = ocr_extract_document(str(pdf_path))
    except Exception as e:
        print(f"[OCR] Skip extra OCR because error: {e!r}")
        return

    # ถ้า ocr_result เป็น dict หรือ object ก็ต้องดึงค่าออกมาให้ถูก
    # สมมติว่า ocr_result มี attribute .texts หรือเป็น dict ที่มี key "texts"
    texts = getattr(ocr_result, "texts", None)
    
    # Fallback ถ้า ocr_result เป็น list ของ TextBlock โดยตรง
    if not texts and isinstance(ocr_result, list):
        texts = ocr_result

    if not texts:
        return

    print(f"[OCR] Attaching {len(texts)} extra OCR pages ...")

    current_index = len(doc.texts)
    doc_id = doc.metadata.doc_id

    for item in texts:
        # รองรับทั้งแบบ Dict และ Object
        if isinstance(item, dict):
            content = (item.get("content") or "").strip()
            page = int(item.get("page") or 1)
        else:
            content = (getattr(item, "content", "") or "").strip()
            page = int(getattr(item, "page", 1))

        if not content:
            continue

        # --- [STEP 1: ลบช่องว่างส่วนเกินที่คั่นระหว่างตัวอักษรไทย] ---
        content = content.replace("\x00", "")
        content = re.sub(r'(?<=[\u0E00-\u0E7F])\s+(?=[\u0E00-\u0E7F])', '', content)

        # --- [STEP 2: ใช้ AI แก้คำผิด (Typo Correction)] ---
        print(f"   🤖 [AI] Correcting typos on page {page}...")
        content = _correct_ocr_text_with_llm(content)
        # -----------------------------------------------------------

        current_index += 1
        block_id = f"ocr_{current_index:04d}"

        tb = TextBlock(
            id=block_id,
            doc_id=doc_id,
            page=page,
            content=content,
            extra={"source": "gemini_ocr_corrected"}, # อัปเดต Source ให้รู้ว่าผ่านการคลีนแล้ว
        )
        doc.texts.append(tb)


def run_ingestion_pipeline(
    pdf_path: str | Path,
    doc_type: str = "generic",
    doc_id: Optional[str] = None,
    output_root: str | Path = "ingested",
    use_ocr: bool = True,
    ocr_mode: str = "auto",
) -> None:
    """
    Ingestion pipeline (Hybrid Version):
    1) เตรียม Folder และ Config
    2) อ่าน PDF ด้วย DoclingParser (Text + Complex Tables as Images)
    3) เสริม OCR (Optional) + AI Corrector
    4) Classify และดึงรูปประกอบทั่วไป
    5) Validate และ Save JSON + Markdown
    """
    pdf_path = Path(pdf_path).resolve()
    output_root = Path(output_root).resolve()

    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF file not found: {pdf_path}")

    # 1) เตรียม Folder ปลายทาง (ต้องทำก่อนเพื่อส่ง path ให้ Docling)
    if not doc_id:
        doc_id = pdf_path.stem.replace(" ", "_")
    
    # โฟลเดอร์ปลายทาง: ingested/{doc_id}/
    doc_dir = output_root / doc_id
    doc_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"=== [INGEST] Start Pipeline for: {doc_id} ===")
    print(f"    PDF: {pdf_path}")
    print(f"    Output: {doc_dir}")

    # สร้าง Config ที่ระบุ output_dir เพื่อส่งให้ Docling
    # นี่คือจุดสำคัญที่ทำให้ Docling รู้ว่าจะเซฟรูปตารางลงที่ไหน
    config = IngestionConfig(output_dir=str(doc_dir))

    # 2) Parse PDF ด้วย Docling
    print(f"[INGEST] Parsing PDF with Docling: {pdf_path}")
    
    try:
        parser = DoclingParser(config=config)
        doc = parser.parse(str(pdf_path), ocr_mode=ocr_mode)
    except Exception as e:
        print(f"[ERROR] Docling parsing failed: {e}")
        return
    
    # --- [Sync Metadata & IDs] ---
    # อัปเดต Metadata พื้นฐาน และ Sync ID ไปทุกส่วน
    doc.metadata.doc_id = doc_id 
    doc.metadata.doc_type = doc_type
    
    # วนลูปยัด ID ใส่ทุกก้อนข้อมูล เพื่อความชัวร์ 100%
    for t in doc.texts: 
        t.doc_id = doc_id
    for tb in doc.tables: 
        tb.doc_id = doc_id
    # -----------------------

    # 3) ต่อข้อความจาก OCR เสริม (ถ้าเปิด) พร้อมผ่านระบบ AI แก้คำผิด
    _attach_ocr_text(doc, pdf_path, use_ocr=use_ocr)

    # 4) Classify doc_type
    if doc_type == "generic" or not doc.metadata.doc_type:
        try:
            detected_type = classify_document(doc, use_llm=False)
            print(f"[INGEST] Detected doc_type: {detected_type}")
            doc.metadata.doc_type = detected_type
        except Exception as e:
            print(f"[WARN] Classification failed: {e}")

    # (Skip Table Extraction: เพราะ Docling ทำให้แล้ว)

    # 5) ดึงรูปประกอบอื่นๆ (ที่ไม่ใช่ตาราง)
    print("[INGEST] Extracting general images ...")
    try:
        # extract_images จะไปเรียก DoclingImageParser หรือวิธีการอื่น
        general_images = extract_images(
            file_path=str(pdf_path), 
            doc_id=doc_id,
            output_root=str(output_root), # ส่ง root ไป เดี๋ยวข้างในไปต่อ path เอง
        )
        if general_images:
            print(f"   + Found {len(general_images)} general images")
            doc.images.extend(general_images)
    except Exception as e:
        print(f"[WARN] Image extraction failed: {e}")

    # 6) Auto-Detect Q&A Pattern
    all_text_content = "\n".join([t.content or "" for t in doc.texts])
    _qna_check_re = re.compile(
        r"(?:ถาม|q|question)\s*[:\-].+?(?:ตอบ|a|answer)\s*[:\-]", 
        re.IGNORECASE | re.DOTALL
    )
    if _qna_check_re.search(all_text_content):
        print(f"👉 [INGEST] Auto-Detect: Found Q&A pattern. Force doc_type='qna'")
        doc.metadata.doc_type = "qna"

    # 7) Validate
    print("[INGEST] Validating document ...")
    issues = validate_all(doc)
    if issues:
        print(f"[WARN] Validation found {len(issues)} issues")

    # 8) Save Output
    metadata_path = doc_dir / "metadata.json"
    text_path = doc_dir / "text.json"
    table_path = doc_dir / "table.json"
    image_path = doc_dir / "image.json"
    validation_path = doc_dir / "validation.json"
    
    # [IMPORTANT] กำหนด Path สำหรับไฟล์ Markdown
    md_path = doc_dir / f"{doc_id}.md"

    # Save JSON files
    with metadata_path.open("w", encoding="utf-8") as f:
        json.dump(doc.metadata.to_dict(), f, ensure_ascii=False, indent=2)

    with text_path.open("w", encoding="utf-8") as f:
        json.dump([t.to_dict() for t in doc.texts], f, ensure_ascii=False, indent=2)

    with table_path.open("w", encoding="utf-8") as f:
        json.dump([tb.to_dict() for tb in doc.tables], f, ensure_ascii=False, indent=2)

    with image_path.open("w", encoding="utf-8") as f:
        json.dump([im.to_dict() for im in doc.images], f, ensure_ascii=False, indent=2)

    with validation_path.open("w", encoding="utf-8") as f:
        json.dump(issues, f, ensure_ascii=False, indent=2)

    # 9) Generate & Save Markdown (.md)
    try:
        print(f"[INGEST] Generating Markdown for {doc_id}...")
        
        # เรียกใช้โดยส่งแค่ doc (ไม่ต้องส่ง output_dir แล้ว)
        md_content = generate_markdown(doc)
        
        # เขียนไฟล์ตรงนี้แทน
        with md_path.open("w", encoding="utf-8") as f:
            f.write(md_content)
            
        print(f"   ✅ Markdown saved to: {md_path}")
    except Exception as e:
        print(f"[WARN] Failed to generate markdown: {e}")

    print("[INGEST] Saved successfully:")
    print(f"  - {metadata_path}")
    print(f"  - {text_path}")
    print(f"  - {table_path} (Tables: {len(doc.tables)})")
    print(f"  - {image_path}")
    print(f"  - {validation_path}")
    print(f"  - {md_path}") 


def main() -> None:
    parser = argparse.ArgumentParser(description="Run ingestion pipeline for a PDF.")
    parser.add_argument("pdf_path", help="Path to PDF file")
    parser.add_argument("--doc-id", default=None, help="Document ID (default: from file_name)")
    parser.add_argument(
        "--doc-type",
        default="generic",
        help="Document type (e.g., bank_statement, invoice, receipt)",
    )
    parser.add_argument(
        "--output-root",
        default="ingested",
        help="Root folder to save ingested outputs (default: 'ingested')",
    )
    parser.add_argument(
        "--no-ocr",
        action="store_true",
        help="Disable extra OCR (Docling has built-in OCR)",
    )
    parser.add_argument(
        "--ocr-mode",
        default="auto",
        choices=["auto", "local", "api"],
        help="OCR strategy: auto (default hybrid), local (Docling only), api (require Vision API)",
    )
    args = parser.parse_args()

    run_ingestion_pipeline(
        pdf_path=args.pdf_path,
        doc_type=args.doc_type,
        doc_id=args.doc_id,
        output_root=args.output_root,
        use_ocr=not args.no_ocr,
        ocr_mode=args.ocr_mode,
    )


if __name__ == "__main__":
    main()