from __future__ import annotations
from ingestion.validator import validate_all
from ingestion.ocr_extractor import ocr_extract_document


"""
run_ingestion.py

สคริปต์หลักสำหรับรัน ingestion เต็ม pipeline:
1) อ่าน PDF ด้วย pdf_parser.parse_pdf -> ได้ metadata + text blocks
2) ให้ AI (Gemini) ช่วยเดาประเภทเอกสาร (bank_statement / invoice / receipt / ...)
3) ดึงตารางด้วย table_extractor.extract_tables -> TableBlock list
4) ดึงรูปด้วย image_extractor.extract_images -> ImageBlock list
5) รวมทั้งหมดเข้า IngestedDocument
6) เซฟออกเป็น JSON แยกไฟล์ในโฟลเดอร์ ingested/{doc_id}/

วิธีรันตัวอย่าง:

    python -m scripts.run_ingestion samples/statement/sample.pdf --doc-type generic

"""

import argparse
import json
from pathlib import Path

from ingestion.pdf_parser import parse_pdf
from ingestion.table_extractor import extract_tables
from ingestion.image_extractor import extract_images
from ingestion.schema import IngestedDocument
from ingestion.document_classifier import classify_document


def save_ingested_document(
    doc: IngestedDocument,
    output_root: str | Path = "ingested",
) -> None:
    """
    เซฟ IngestedDocument ลงเป็นไฟล์ JSON แยกตามประเภท:

    - ingested/{doc_id}/metadata.json
    - ingested/{doc_id}/text.json
    - ingested/{doc_id}/table.json
    - ingested/{doc_id}/image.json
    """
    output_root = Path(output_root)
    doc_id = doc.metadata.doc_id

    doc_dir = output_root / doc_id
    doc_dir.mkdir(parents=True, exist_ok=True)

    # 1) metadata.json
    metadata_path = doc_dir / "metadata.json"
    with metadata_path.open("w", encoding="utf-8") as f:
        json.dump(doc.metadata.to_dict(), f, ensure_ascii=False, indent=2)

    # 2) text.json
    text_path = doc_dir / "text.json"
    with text_path.open("w", encoding="utf-8") as f:
        json.dump(
            [t.to_dict() for t in doc.texts],
            f,
            ensure_ascii=False,
            indent=2,
        )

    # 3) table.json
    table_path = doc_dir / "table.json"
    with table_path.open("w", encoding="utf-8") as f:
        json.dump(
            [tb.to_dict() for tb in doc.tables],
            f,
            ensure_ascii=False,
            indent=2,
        )

    # 4) image.json
    image_path = doc_dir / "image.json"
    with image_path.open("w", encoding="utf-8") as f:
        json.dump(
            [im.to_dict() for im in doc.images],
            f,
            ensure_ascii=False,
            indent=2,
        )

    print(f"[run_ingestion] Saved metadata to: {metadata_path}")
    print(f"[run_ingestion] Saved texts to:    {text_path}")
    print(f"[run_ingestion] Saved tables to:   {table_path}")
    print(f"[run_ingestion] Saved images to:   {image_path}")


def run_ingestion_pipeline(
    pdf_path: str | Path,
    doc_type: str = "generic",
    doc_id: str | None = None,
    output_root: str | Path = "ingested",
) -> IngestedDocument:
    """
    รัน ingestion ครบชุดสำหรับ PDF 1 ไฟล์:

    - parse_pdf -> metadata + texts
    - ใช้ Gemini ช่วย classify ประเภทเอกสาร
    - extract_tables -> tables
    - extract_images -> images
    - รวมทั้งหมดกลับเข้า IngestedDocument
    - เซฟ JSON ลงโฟลเดอร์

    :return: IngestedDocument
    """
    pdf_path = Path(pdf_path)

    # 1) parse PDF เพื่อให้ได้ metadata + texts
    print(f"[run_ingestion] Parsing PDF text from: {pdf_path}")
    doc = parse_pdf(
        file_path=pdf_path,
        doc_type=doc_type,  # initial hint เผื่ออยากส่ง
        doc_id=doc_id,
        source="uploaded",
    )

    # doc_id ที่ใช้จริง (มาจาก metadata เสมอ)
    effective_doc_id = doc.metadata.doc_id

    # 2) ให้ Gemini ช่วยฟันธงประเภทเอกสาร
    try:
        predicted_type = classify_document(doc, use_gemini=True)
        print(f"[run_ingestion] Predicted document type: {predicted_type}")
        doc.metadata.doc_type = predicted_type
    except Exception as e:
        print(f"[run_ingestion] Document classification failed: {e}")
        print("[run_ingestion] Keep original doc_type:", doc.metadata.doc_type)

    # 3) extract tables
    print(f"[run_ingestion] Extracting tables for doc_id={effective_doc_id}")
    tables = extract_tables(
        file_path=pdf_path,
        doc_id=effective_doc_id,
        doc_type=doc.metadata.doc_type,
        pages="all",
    )
    doc.tables = tables

    # 4) extract images
    print(f"[run_ingestion] Extracting images for doc_id={effective_doc_id}")
    images = extract_images(
        file_path=pdf_path,
        doc_id=effective_doc_id,
        output_root=output_root,
    )
    doc.images = images
    
    # 4) run validation
    print(f"[run_ingestion] Validating document for doc_id={effective_doc_id}")
    issues = validate_all(doc)

    # เซฟ validation.json
    output_root = Path(output_root)
    doc_dir = output_root / effective_doc_id
    doc_dir.mkdir(parents=True, exist_ok=True)
    validation_path = doc_dir / "validation.json"

    with validation_path.open("w", encoding="utf-8") as f:
        json.dump(issues, f, ensure_ascii=False, indent=2)

    print(f"[run_ingestion] Validation issues: {len(issues)} (saved to {validation_path})")

    # 5) save all as JSON files
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
    parser.add_argument(
        "--doc-type",
        default="generic",
        help="Document type hint (e.g., bank_statement, receipt, invoice)",
    )
    parser.add_argument(
        "--doc-id",
        default=None,
        help="Override document ID (default: stem of file name)",
    )
    parser.add_argument(
        "--output-root",
        default="ingested",
        help="Root folder to save ingested outputs (default: 'ingested')",
    )
    args = parser.parse_args()

    run_ingestion_pipeline(
        pdf_path=args.pdf_path,
        doc_type=args.doc_type,
        doc_id=args.doc_id,
        output_root=args.output_root,
    )


if __name__ == "__main__":
    main()
