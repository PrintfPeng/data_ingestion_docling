# scripts/run_ingestion.py
from __future__ import annotations

from pathlib import Path
from typing import Optional
import json

from ingestion.pdf_parser import parse_pdf
from ingestion.document_classifier import classify_document
from ingestion.table_extractor import extract_tables
from ingestion.image_extractor import extract_images
from ingestion.schema import IngestedDocument, TextBlock
from ingestion.validator import validate_all
from ingestion.ocr_extractor import ocr_extract_document


def _attach_ocr_text(doc: IngestedDocument, pdf_path: Path, use_ocr: bool = True) -> None:
    """
    เรียก OCR ด้วย Gemini แล้วเอาข้อความต่อเข้าไปใน doc.texts

    - ถ้า use_ocr=False จะไม่ทำอะไร
    - ข้อความจาก OCR จะถูกใส่เป็น TextBlock ใหม่ ๆ
      พร้อม extra={"source": "ocr"}
    """
    if not use_ocr:
        return

    try:
        ocr_result = ocr_extract_document(str(pdf_path))
    except Exception as e:  # noqa: BLE001
        print(f"[OCR] Skip OCR because error: {e!r}")
        return

    texts = getattr(ocr_result, "texts", None)
    if not texts:
        print("[OCR] No OCR texts found.")
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


def run_ingestion_pipeline(
    pdf_path: str | Path,
    doc_type: str = "generic",
    doc_id: Optional[str] = None,
    output_root: str | Path = "ingested",
    use_ocr: bool = True,       # <-- เปิด OCR เป็นค่า default
) -> None:
    """
    Ingestion pipeline หลัก:
    1) อ่าน PDF เป็น text ปกติ (pdf_parser)
    2) ต่อข้อความจาก OCR (ถ้าเปิด use_ocr)
    3) classify ประเภทเอกสาร
    4) ดึงตาราง
    5) ดึงรูป
    6) validate แล้วเซฟ JSON ลง ingested/{doc_id}/
    """
    pdf_path = Path(pdf_path)
    output_root = Path(output_root)

    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF file not found: {pdf_path}")

    # 1) parse PDF เป็น text ปกติ
    print(f"[INGEST] Parsing PDF: {pdf_path}")
    doc = parse_pdf(
        file_path=pdf_path,
        doc_type=doc_type,
        doc_id=doc_id,
        source="uploaded",
    )  # type: IngestedDocument

    # 2) ต่อข้อความจาก OCR
    _attach_ocr_text(doc, pdf_path, use_ocr=use_ocr)

    # 3) classify doc_type (ถ้ายัง generic)
    if doc_type == "generic" or not doc.metadata.doc_type:
        detected_type = classify_document(doc, use_gemini=False)
        print(f"[INGEST] Detected doc_type: {detected_type}")
        doc.metadata.doc_type = detected_type
    else:
        doc.metadata.doc_type = doc_type

    doc_id = doc.metadata.doc_id

    # 4) ดึงตาราง
    print("[INGEST] Extracting tables ...")
    tables = extract_tables(
        file_path=pdf_path,
        doc_id=doc_id,
        doc_type=doc.metadata.doc_type,
    )
    doc.tables = tables

    # 5) ดึงรูป
    print("[INGEST] Extracting images ...")
    images = extract_images(
        file_path=pdf_path,
        doc_id=doc_id,
        output_root=output_root,
    )
    doc.images = images

    # 6) validate
    print("[INGEST] Validating document ...")
    issues = validate_all(doc)

    # 7) เซฟเป็น JSON
    doc_dir = output_root / doc_id
    doc_dir.mkdir(parents=True, exist_ok=True)

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
        json.dump([tb.to_dict() for tb in doc.tables], f, ensure_ascii=False, indent=2)

    with image_path.open("w", encoding="utf-8") as f:
        json.dump([im.to_dict() for im in doc.images], f, ensure_ascii=False, indent=2)

    with validation_path.open("w", encoding="utf-8") as f:
        json.dump(issues, f, ensure_ascii=False, indent=2)

    print("[INGEST] Saved:")
    print(f"  - {metadata_path}")
    print(f"  - {text_path}")
    print(f"  - {table_path}")
    print(f"  - {image_path}")
    print(f"  - {validation_path}")


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Run ingestion pipeline for a PDF.")
    parser.add_argument("pdf_path", help="Path to PDF file")
    parser.add_argument("--doc-id", default=None, help="Document ID (default: from filename)")
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
        help="Disable OCR (by default OCR is enabled)",
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
