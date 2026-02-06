# scripts/run_ingestion.py
from __future__ import annotations

from pathlib import Path
from typing import Optional
import json
import re

# [CHANGE] ใช้ DoclingParser ตัวใหม่ที่เราทำ Hybrid Ingestion ไว้
from ingestion.docling_parser import DoclingParser
# from ingestion.pdf_parser import parse_pdf  <-- ไม่ใช้ตัวเก่าแล้ว
# from ingestion.table_extractor import extract_tables <-- ไม่ใช้แล้วเพราะ Docling จัดการให้

from ingestion.document_classifier import classify_document
from ingestion.image_extractor import extract_images
from ingestion.schema import IngestedDocument, TextBlock
from ingestion.validator import validate_all
from ingestion.ocr_extractor import ocr_extract_document


# Helper Class สำหรับส่ง Config ให้ DoclingParser (เพื่อให้เซฟรูปได้ถูกที่)
class IngestionConfig:
    def __init__(self, output_dir: str):
        self.output_dir = output_dir


def _attach_ocr_text(doc: IngestedDocument, pdf_path: Path, use_ocr: bool = True) -> None:
    """
    เรียก OCR ด้วย Gemini แล้วเอาข้อความต่อเข้าไปใน doc.texts
    """
    if not use_ocr:
        return

    try:
        # หมายเหตุ: Docling มี OCR ในตัวอยู่แล้ว (Tesseract)
        # แต่ถ้าอยากใช้ Gemini OCR เสริมอีก ก็เปิดไว้ได้ครับ
        ocr_result = ocr_extract_document(str(pdf_path))
    except Exception as e:
        print(f"[OCR] Skip extra OCR because error: {e!r}")
        return

    texts = getattr(ocr_result, "texts", None)
    if not texts:
        return

    print(f"[OCR] Attaching {len(texts)} extra OCR pages ...")

    current_index = len(doc.texts)
    doc_id = doc.metadata.doc_id

    for item in texts:
        content = (item.get("content") or "").strip()
        if not content:
            continue

        page = int(item.get("page") or 1)
        current_index += 1
        block_id = f"ocr_{current_index:04d}"

        tb = TextBlock(
            id=block_id,
            doc_id=doc_id,
            page=page,
            content=content,
            extra={"source": "gemini_ocr"}, # ระบุ source ให้ชัดเจน
        )
        doc.texts.append(tb)


def run_ingestion_pipeline(
    pdf_path: str | Path,
    doc_type: str = "generic",
    doc_id: Optional[str] = None,
    output_root: str | Path = "ingested",
    use_ocr: bool = True,
) -> None:
    """
    Ingestion pipeline (Hybrid Version):
    1) เตรียม Folder และ Config
    2) อ่าน PDF ด้วย DoclingParser (Text + Complex Tables as Images)
    3) เสริม OCR (Optional)
    4) Classify และดึงรูปประกอบทั่วไป
    5) Validate และ Save
    """
    pdf_path = Path(pdf_path)
    output_root = Path(output_root)

    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF file not found: {pdf_path}")

    # 1) เตรียม Folder ปลายทาง (ต้องทำก่อนเพื่อส่ง path ให้ Docling)
    if not doc_id:
        doc_id = pdf_path.stem.replace(" ", "_")
    
    doc_dir = output_root / doc_id
    doc_dir.mkdir(parents=True, exist_ok=True)

    # สร้าง Config ที่ระบุ output_dir
    config = IngestionConfig(output_dir=str(doc_dir))

# 2) Parse PDF ด้วย Docling
    print(f"[INGEST] Parsing PDF with Docling: {pdf_path}")
    parser = DoclingParser(config=config)
    doc = parser.parse(str(pdf_path))
    
    # --- [เพิ่มตรงนี้ครับ] ---
    # อัปเดต Metadata พื้นฐาน และ Sync ID ไปทุกส่วน
    doc.metadata.doc_id = doc_id 
    doc.metadata.doc_type = doc_type
    
    # วนลูปยัด ID ใส่ทุกก้อนข้อมูล เพื่อความชัวร์ 100%
    for t in doc.texts: 
        t.doc_id = doc_id
    for tb in doc.tables: 
        tb.doc_id = doc_id
    # -----------------------

    # 3) ต่อข้อความจาก OCR เสริม (ถ้าเปิด)
    _attach_ocr_text(doc, pdf_path, use_ocr=use_ocr)

    # 4) Classify doc_type
    if doc_type == "generic" or not doc.metadata.doc_type:
        detected_type = classify_document(doc, use_llm=False)
        print(f"[INGEST] Detected doc_type: {detected_type}")
        doc.metadata.doc_type = detected_type

    # (Skip Table Extraction: เพราะ Docling ทำให้แล้ว)

    # 5) ดึงรูปประกอบอื่นๆ (ที่ไม่ใช่ตาราง)
    print("[INGEST] Extracting general images ...")
    try:
        general_images = extract_images(
            file_path=pdf_path,
            doc_id=doc_id,
            output_root=output_root,
        )
        # เติมเข้าไป (ต่อท้ายรูปตารางที่มีอยู่แล้ว)
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

    # 8) Save Output
    metadata_path = doc_dir / "metadata.json"
    text_path = doc_dir / "text.json"
    table_path = doc_dir / "table.json"
    image_path = doc_dir / "image.json"
    validation_path = doc_dir / "validation.json"

    with metadata_path.open("w", encoding="utf-8") as f:
        json.dump(doc.metadata.to_dict(), f, ensure_ascii=False, indent=2)

    with text_path.open("w", encoding="utf-8") as f:
        json.dump([t.to_dict() for t in doc.texts], f, ensure_ascii=False, indent=2)

    with table_path.open("w", encoding="utf-8") as f:
        # ตรงนี้สำคัญ: to_dict() จะรวม image_path ไปด้วย
        json.dump([tb.to_dict() for tb in doc.tables], f, ensure_ascii=False, indent=2)

    with image_path.open("w", encoding="utf-8") as f:
        json.dump([im.to_dict() for im in doc.images], f, ensure_ascii=False, indent=2)

    with validation_path.open("w", encoding="utf-8") as f:
        json.dump(issues, f, ensure_ascii=False, indent=2)

    print("[INGEST] Saved successfully:")
    print(f"  - {metadata_path}")
    print(f"  - {text_path}")
    print(f"  - {table_path} (Tables: {len(doc.tables)})")
    print(f"  - {image_path}")
    print(f"  - {validation_path}")


def main() -> None:
    import argparse

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
    args = parser.parse_args()

    run_ingestion_pipeline(
        pdf_path=args.pdf_path,
        doc_type=args.doc_type,
        doc_id=args.doc_id,
        output_root=args.output_root,
        use_ocr=not args.no_ocr,
    )


if __name__ == "__main__":
    main()