from __future__ import annotations

"""
run_ingestion.py (Updated with OCR)
"""

import argparse
import json
from pathlib import Path

from ingestion.pdf_parser import parse_pdf
from ingestion.table_extractor import extract_tables
from ingestion.image_extractor import extract_images
from ingestion.schema import IngestedDocument, TextBlock # <--- เพิ่ม TextBlock
from ingestion.document_classifier import classify_document
from ingestion.validator import validate_all
from ingestion.ocr_extractor import ocr_extract_document # <--- เพิ่ม import นี้


def _attach_ocr_text(doc: IngestedDocument, pdf_path: Path) -> None:
    """
    ฟังก์ชันเสริม: เรียก OCR แล้วเอาข้อความมาต่อท้ายใน doc.texts
    """
    try:
        # เรียก OCR (มันจะ Auto-detect หน้าที่เป็นรูปภาพให้เองตาม Logic ใหม่ที่เราแก้)
        ocr_result = ocr_extract_document(str(pdf_path))
    except Exception as e:
        print(f"[OCR] Skip OCR because error: {e}")
        return

    texts = getattr(ocr_result, "texts", None)
    if not texts:
        print("[OCR] No OCR texts found (or API failed).")
        return

    print(f"[OCR] Attaching {len(texts)} OCR pages to text blocks ...")

    current_index = len(doc.texts)
    doc_id = doc.metadata.doc_id

    for item in texts:
        content = (item.get("content") or "").strip()
        if not content:
            continue

        page = int(item.get("page") or 1)
        current_index += 1
        block_id = f"ocr_{current_index:04d}"

        # สร้าง TextBlock ใหม่จากผล OCR
        tb = TextBlock(
            id=block_id,
            doc_id=doc_id,
            page=page,
            content=content,
            section=None,
            category=None,
            bbox=None,
            extra={"source": "ocr"},
        )
        doc.texts.append(tb)


def save_ingested_document(
    doc: IngestedDocument,
    output_root: str | Path = "ingested",
) -> None:
    output_root = Path(output_root)
    doc_id = doc.metadata.doc_id

    doc_dir = output_root / doc_id
    doc_dir.mkdir(parents=True, exist_ok=True)

    with (doc_dir / "metadata.json").open("w", encoding="utf-8") as f:
        json.dump(doc.metadata.to_dict(), f, ensure_ascii=False, indent=2)

    with (doc_dir / "text.json").open("w", encoding="utf-8") as f:
        json.dump([t.to_dict() for t in doc.texts], f, ensure_ascii=False, indent=2)

    with (doc_dir / "table.json").open("w", encoding="utf-8") as f:
        json.dump([tb.to_dict() for tb in doc.tables], f, ensure_ascii=False, indent=2)

    with (doc_dir / "image.json").open("w", encoding="utf-8") as f:
        json.dump([im.to_dict() for im in doc.images], f, ensure_ascii=False, indent=2)
        
    print(f"[run_ingestion] Saved output files to: {doc_dir}")


def run_ingestion_pipeline(
    pdf_path: str | Path,
    doc_type: str = "generic",
    doc_id: str | None = None,
    output_root: str | Path = "ingested",
) -> IngestedDocument:
    
    pdf_path = Path(pdf_path)

    # 1) Parse PDF (Text layer)
    print(f"[run_ingestion] Parsing PDF text from: {pdf_path}")
    doc = parse_pdf(
        file_path=pdf_path,
        doc_type=doc_type,
        doc_id=doc_id,
        source="uploaded",
    )

    # ---------------------------------------------------------
    # 2) [NEW] เรียก OCR เสมอ (Logic ข้างในจะเช็คเองว่าต้องทำไหม)
    # ---------------------------------------------------------
    _attach_ocr_text(doc, pdf_path)
    # ---------------------------------------------------------

    effective_doc_id = doc.metadata.doc_id

    # 3) Classify Type
    try:
        predicted_type = classify_document(doc, use_llm=True)
        print(f"[run_ingestion] Predicted document type: {predicted_type}")
        doc.metadata.doc_type = predicted_type
    except Exception as e:
        print(f"[run_ingestion] Document classification warning: {e}")

    # 4) Extract Tables
    print(f"[run_ingestion] Extracting tables for doc_id={effective_doc_id}")
    tables = extract_tables(
        file_path=pdf_path,
        doc_id=effective_doc_id,
        doc_type=doc.metadata.doc_type,
        pages="all",
    )
    doc.tables = tables

    # 5) Extract Images
    print(f"[run_ingestion] Extracting images for doc_id={effective_doc_id}")
    images = extract_images(
        file_path=pdf_path,
        doc_id=effective_doc_id,
        output_root=output_root,
    )
    doc.images = images
    
    # 6) Validation
    print(f"[run_ingestion] Validating document for doc_id={effective_doc_id}")
    issues = validate_all(doc)
    
    doc_dir = Path(output_root) / effective_doc_id
    doc_dir.mkdir(parents=True, exist_ok=True)
    with (doc_dir / "validation.json").open("w", encoding="utf-8") as f:
        json.dump(issues, f, ensure_ascii=False, indent=2)

    # 7) Save
    print(f"[run_ingestion] Saving ingested document for doc_id={effective_doc_id}")
    save_ingested_document(doc, output_root=output_root)

    print(
        f"[run_ingestion] Done. Texts={len(doc.texts)}, "
        f"Tables={len(doc.tables)}, Images={len(doc.images)}",
    )
    return doc


def main() -> None:
    parser = argparse.ArgumentParser(description="Run full PDF ingestion pipeline.")
    parser.add_argument("pdf_path", help="Path to PDF file")
    parser.add_argument("--doc-type", default="generic", help="Document type hint")
    parser.add_argument("--doc-id", default=None, help="Override document ID")
    parser.add_argument("--output-root", default="ingested", help="Root folder to save outputs")
    
    args = parser.parse_args()

    run_ingestion_pipeline(
        pdf_path=args.pdf_path,
        doc_type=args.doc_type,
        doc_id=args.doc_id,
        output_root=args.output_root,
    )

if __name__ == "__main__":
    main()