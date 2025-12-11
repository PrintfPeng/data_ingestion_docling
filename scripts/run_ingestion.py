from __future__ import annotations

import json
import logging
from pathlib import Path
import sys
import argparse

# Add project root to path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from ingestion.pdf_parser import parse_pdf
from ingestion.table_extractor import extract_tables
from ingestion.image_extractor import extract_images
from ingestion.ocr_extractor import ocr_extract_document
from ingestion.cleaner import clean_text_blocks, clean_table_blocks, remove_text_inside_tables
from ingestion.document_classifier import classify_document
from ingestion.semantic_enricher import (
    tag_sections, 
    categorize_text_blocks, 
    normalize_tables,
    prepare_mapping_payload
)
from ingestion.schema import TextBlock, DocumentMetadata, IngestedDocument

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def ingest_single_file(
    pdf_path: str, 
    output_root: str = "ingested",
    doc_id: str = None,
    doc_type: str = "generic",
):
    path = Path(pdf_path)
    if not doc_id: doc_id = path.stem

    output_dir = Path(output_root) / doc_id
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"=== Starting Ingestion (Force OCR Mode): {doc_id} ===")

    # 1. Parse Structure Only (ใช้ parse_pdf เพื่อดึง Metadata จำนวนหน้า แต่ไม่เอา Text)
    logger.info("Step 1: Parsing PDF Structure...")
    doc = parse_pdf(path, doc_id=doc_id, doc_type=doc_type)
    
    # ⚠️ CRITICAL FIX: ล้าง Text เดิมทิ้งทั้งหมด เพราะมันเป็นภาษาต่างดาว
    doc.texts = [] 
    logger.info("        Cleared garbled native text. Preparing for Full OCR...")

    # 2. Force OCR on ALL Pages
    total_pages = doc.metadata.page_count
    all_pages = set(range(1, total_pages + 1)) # สร้างลิสต์หน้า 1 ถึงหน้าสุดท้าย
    
    logger.info(f"Step 2: Running OCR on all {total_pages} pages via API...")
    
    # ส่งทุกหน้าไปทำ OCR
    ocr_result = ocr_extract_document(str(path), target_pages=all_pages)
    
    # เอาผลลัพธ์จาก OCR มาใส่ใน doc.texts
    ocr_count = 0
    for item in ocr_result.texts:
        # สร้าง Block ข้อความใหม่จาก OCR
        # หมายเหตุ: OCR แบบนี้อาจจะไม่มี BBox (พิกัด) ที่แม่นยำ แต่จะได้ข้อความที่อ่านออก
        new_block = TextBlock(
            id=f"ocr_p{item['page']:02d}_{ocr_count}",
            doc_id=doc_id,
            page=item['page'],
            content=item['content'],
            bbox=[0, 0, 0, 0], # ใส่ Dummy bbox ไปก่อน
            extra={"source": "ocr_api_tesseract"}
        )
        doc.texts.append(new_block)
        ocr_count += 1
        
    logger.info(f"        OCR Completed. Captured {ocr_count} text blocks.")

    # 3. Extract Tables
    logger.info("Step 3: Extracting Tables...")
    try:
        tables = extract_tables(path, doc_id=doc_id)
        doc.tables = tables
    except Exception as e:
        logger.warning(f"Table extraction failed: {e}")
        doc.tables = []

    # 4. Extract Images
    logger.info("Step 4: Extracting Images...")
    try:
        images = extract_images(path, doc_id=doc_id, output_root=output_root)
        doc.images = images
    except Exception as e:
        logger.warning(f"Image extraction failed: {e}")
        doc.images = []

    # 5. Cleaning
    logger.info("Step 5: Cleaning Data...")
    # ข้าม remove_text_inside_tables เพราะ bbox ของ OCR เราไม่แม่นยำ
    # doc.texts = remove_text_inside_tables(doc.texts, doc.tables) 
    doc.tables = clean_table_blocks(doc.tables)
    doc.texts = clean_text_blocks(doc.texts)

    # 6. Enrich
    logger.info("Step 6: Semantic Enrichment...")
    if doc_type == "generic" or not doc_type:
        doc_type = classify_document(doc, use_gemini=False)
    doc.metadata.doc_type = doc_type
    doc = tag_sections(doc, use_gemini=False)
    doc = categorize_text_blocks(doc, use_gemini=False)
    doc.tables = normalize_tables(doc.tables)

    # 7. Save
    logger.info(f"Step 7: Saving to {output_dir} ...")
    (output_dir / "metadata.json").write_text(json.dumps(doc.metadata.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")
    (output_dir / "text_clean.json").write_text(json.dumps([t.to_dict() for t in doc.texts], ensure_ascii=False, indent=2), encoding="utf-8")
    (output_dir / "table_normalized.json").write_text(json.dumps([t.to_dict() for t in doc.tables], ensure_ascii=False, indent=2), encoding="utf-8")
    (output_dir / "image.json").write_text(json.dumps([im.to_dict() for im in doc.images], ensure_ascii=False, indent=2), encoding="utf-8")
    (output_dir / "mapping.json").write_text(json.dumps(prepare_mapping_payload(doc), ensure_ascii=False, indent=2), encoding="utf-8")

    logger.info("=== Ingestion Completed Successfully ===")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("pdf_path", help="Path to PDF file")
    parser.add_argument("--doc-id", help="Document ID to use", default=None)
    parser.add_argument("--doc-type", help="Document Type", default="generic")
    args = parser.parse_args()
    
    ingest_single_file(args.pdf_path, doc_id=args.doc_id, doc_type=args.doc_type)