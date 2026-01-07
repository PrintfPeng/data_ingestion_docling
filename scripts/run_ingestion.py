from __future__ import annotations

import argparse
import json
import time
from datetime import datetime
from pathlib import Path
import pandas as pd

# Import Schema ของเรา
from ingestion.schema import IngestedDocument, TextBlock, TableBlock, ImageBlock, DocumentMetadata

# Import Docling Parser
from ingestion.docling_parser import DoclingParser
from ingestion.document_classifier import classify_document
from ingestion.validator import validate_all

# Import Semantic Enricher
from ingestion.semantic_enricher import tag_sections, categorize_text_blocks, normalize_tables, prepare_mapping_payload


def docling_to_ingested_doc(doc_result: dict, doc_id: str, file_path: str) -> IngestedDocument:
    """
    แปลง Output จาก Docling ให้เป็น IngestedDocument Schema ของเรา
    """
    doc = doc_result["doc"]
    saved_images = doc_result["saved_images"]
    
    # -------------------------------------------------------
    # 1. Texts
    # -------------------------------------------------------
    text_blocks = []
    counter = 0
    
    # Docling v2: doc.texts เก็บ list ของ TextItem
    if hasattr(doc, "texts"):
        for item in doc.texts:
            content = item.text.strip()
            if not content:
                continue
            
            counter += 1
            # พยายามดึงเลขหน้า
            page = 1
            bbox = None
            if hasattr(item, "prov") and item.prov:
                # prov อาจจะเป็น list หรือ single object ขึ้นอยู่กับ version
                prov_list = item.prov if isinstance(item.prov, list) else [item.prov]
                if prov_list:
                    first_prov = prov_list[0]
                    if hasattr(first_prov, "page_no"):
                        page = first_prov.page_no
                    if hasattr(first_prov, "bbox") and first_prov.bbox:
                         if hasattr(first_prov.bbox, "as_tuple"):
                             bbox = first_prov.bbox.as_tuple()
            
            text_blocks.append(TextBlock(
                id=f"txt_{counter:04d}",
                doc_id=doc_id,
                page=page,
                content=content,
                bbox=bbox,
                extra={"source": "docling"}
            ))
            
    # Fallback ถ้า doc.texts ไม่มี
    if not text_blocks:
        print("[Warn] Using markdown fallback for text blocks")
        md_text = doc_result.get("markdown", "")
        if md_text:
            md_lines = md_text.splitlines()
            for i, line in enumerate(md_lines):
                if line.strip():
                    text_blocks.append(TextBlock(
                        id=f"txt_md_{i:04d}",
                        doc_id=doc_id,
                        page=1, 
                        content=line.strip(),
                        extra={"source": "docling_markdown"}
                    ))

    # -------------------------------------------------------
    # 2. Tables
    # -------------------------------------------------------
    table_blocks = []
    if hasattr(doc, "tables"):
        for i, tbl in enumerate(doc.tables):
            # [FIXED] แก้เรื่อง Warning และ TypeError
            try:
                # พยายามส่ง doc=doc ตามที่ Docling รุ่นใหม่ต้องการ
                df = tbl.export_to_dataframe(doc=doc)
                html = tbl.export_to_html(doc=doc)
            except Exception:
                # ถ้าพัง (เช่น version ไม่ตรง) ให้ลองแบบไม่ส่ง doc
                try:
                    df = tbl.export_to_dataframe()
                    html = tbl.export_to_html()
                except Exception as e:
                    print(f"[Warn] Table export failed: {e}")
                    continue
            
            if df.empty:
                continue
                
            header = [str(c) for c in df.columns]
            rows = df.astype(str).values.tolist()
            
            page = 1
            bbox = None
            if hasattr(tbl, "prov") and tbl.prov:
                prov_list = tbl.prov if isinstance(tbl.prov, list) else [tbl.prov]
                if prov_list:
                    p = prov_list[0]
                    if hasattr(p, "page_no"): page = p.page_no
                    if hasattr(p, "bbox") and p.bbox:
                        if hasattr(p.bbox, "as_tuple"): bbox = p.bbox.as_tuple()
            
            table_blocks.append(TableBlock(
                id=f"tbl_{i+1:04d}",
                doc_id=doc_id,
                page=page,
                name=f"Table {i+1}",
                category="docling_extracted",
                columns=header,
                rows=rows,
                bbox=bbox,
                extra={
                    "html_content": html,
                    "method": "docling"
                }
            ))

    # -------------------------------------------------------
    # 3. Images
    # -------------------------------------------------------
    image_blocks = []
    for img_meta in saved_images:
        image_blocks.append(ImageBlock(
            id=f"img_{img_meta['index']+1:04d}",
            doc_id=doc_id,
            page=img_meta["page"],
            file_path=img_meta["path"],
            caption="", 
            bbox=img_meta["bbox"],
            extra={"source": "docling"}
        ))

    # -------------------------------------------------------
    # 4. Construct Document
    # -------------------------------------------------------
    
    # [FIXED] แก้ Bug "method is not JSON serializable" ตรงนี้
    page_count = 1
    if hasattr(doc, "num_pages"):
        if callable(doc.num_pages):
            page_count = doc.num_pages() # เรียก () ถ้าเป็น method
        else:
            page_count = int(doc.num_pages) # แปลงเป็น int ถ้าเป็น property
    else:
        # Fallback
        all_pages = [t.page for t in text_blocks] + [i.page for i in image_blocks]
        if all_pages:
            page_count = max(all_pages)

    return IngestedDocument(
        metadata=DocumentMetadata(
            doc_id=doc_id,
            file_name=Path(file_path).name,
            doc_type="generic",
            page_count=page_count,
            ingested_at=datetime.now().isoformat(), 
            source="docling"
        ),
        texts=text_blocks,
        tables=table_blocks,
        images=image_blocks
    )


def save_ingested_document(doc: IngestedDocument, output_root: str | Path):
    output_root = Path(output_root)
    doc_dir = output_root / doc.metadata.doc_id
    doc_dir.mkdir(parents=True, exist_ok=True)

    with (doc_dir / "metadata.json").open("w", encoding="utf-8") as f:
        json.dump(doc.metadata.to_dict(), f, ensure_ascii=False, indent=2)
    with (doc_dir / "text.json").open("w", encoding="utf-8") as f:
        json.dump([t.to_dict() for t in doc.texts], f, ensure_ascii=False, indent=2)
    with (doc_dir / "table.json").open("w", encoding="utf-8") as f:
        json.dump([tb.to_dict() for tb in doc.tables], f, ensure_ascii=False, indent=2)
    with (doc_dir / "image.json").open("w", encoding="utf-8") as f:
        json.dump([im.to_dict() for im in doc.images], f, ensure_ascii=False, indent=2)
    
    print(f"[run_ingestion] Saved raw output to: {doc_dir}")


def run_ingestion_pipeline(
    pdf_path: str | Path,
    doc_type: str = "generic",
    doc_id: str | None = None,
    output_root: str | Path = "ingested",
) -> IngestedDocument:
    
    pdf_path = Path(pdf_path)
    if not doc_id:
        doc_id = pdf_path.stem

    # ----------------------------------------------------
    # 1. USE DOCLING
    # ----------------------------------------------------
    print(f"==== [1/3] Ingestion (Docling) ====")
    image_dir = Path(output_root) / doc_id / "images"
    
    parser = DoclingParser(output_dir=str(Path(output_root)/doc_id), image_dir=str(image_dir))
    
    # Run Docling Parse
    doc_result = parser.parse_file(str(pdf_path))
    
    # 🔥 [NEW] บันทึกไฟล์ Markdown (.md)
    # ----------------------------------------------------
    md_content = doc_result.get("markdown", "")
    if md_content:
        # ตั้งชื่อไฟล์เป็น {doc_id}.md
        md_path = Path(output_root) / doc_id / f"{doc_id}.md"
        # สร้างโฟลเดอร์ให้ชัวร์ว่ามีอยู่จริง
        md_path.parent.mkdir(parents=True, exist_ok=True)
        with open(md_path, "w", encoding="utf-8") as f:
            f.write(md_content)
        print(f"[Docling] Saved Markdown to: {md_path}")
    # ----------------------------------------------------
    
    # Convert to Schema
    doc = docling_to_ingested_doc(doc_result, doc_id, str(pdf_path))
    doc.metadata.doc_type = doc_type

    print(f"[Docling] Extracted: Texts={len(doc.texts)}, Tables={len(doc.tables)}, Images={len(doc.images)}")
    
    save_ingested_document(doc, output_root)

    # ----------------------------------------------------
    # 2. ENRICHMENT
    # ----------------------------------------------------
    print(f"==== [2/3] Enrichment ====")
    
    # Classify Type
    try:
        predicted_type = classify_document(doc, use_gemini=True)
        print(f"[Classifier] Predicted type: {predicted_type}")
        doc.metadata.doc_type = predicted_type
    except Exception as e:
        print(f"[Classifier] Error: {e}")

    # Tag Sections
    doc = tag_sections(doc, use_gemini=True)
    doc = categorize_text_blocks(doc, use_gemini=True)
    
    # Normalize Tables
    doc.tables = normalize_tables(doc.tables)
    
    # Save Enriched Data
    doc_dir = Path(output_root) / doc_id
    with (doc_dir / "text_enriched.json").open("w", encoding="utf-8") as f:
        json.dump([t.to_dict() for t in doc.texts], f, ensure_ascii=False, indent=2)
    with (doc_dir / "table_normalized.json").open("w", encoding="utf-8") as f:
        json.dump([tb.to_dict() for tb in doc.tables], f, ensure_ascii=False, indent=2)
        
    print(f"==== [3/3] Done ====")
    return doc


def main():
    parser = argparse.ArgumentParser(description="Run Docling ingestion pipeline.")
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