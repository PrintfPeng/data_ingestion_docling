from __future__ import annotations

import argparse
import json
import time
from datetime import datetime
from pathlib import Path
import pandas as pd

# Import Schema
from ingestion.schema import IngestedDocument, TextBlock, TableBlock, ImageBlock, DocumentMetadata

# Import Parsers & Helpers
from ingestion.docling_parser import DoclingParser
from ingestion.document_classifier import classify_document
from ingestion.validator import validate_all
from ingestion.semantic_enricher import tag_sections, categorize_text_blocks, normalize_tables
# [FIX] Import ชื่อฟังก์ชันให้ถูกต้อง (เปลี่ยนจาก _get_gemini_vision_model เป็น _get_gemini_vision_client)
from ingestion.image_extractor import _get_gemini_vision_client, generate_image_description_md


def enrich_images_with_context(doc_result: dict, doc_id: str, model) -> list[ImageBlock]:
    """
    ฟังก์ชันผูกบริบท Text รอบข้างเข้ากับรูปภาพ และให้ Gemini เขียนคำบรรยาย
    ทำทีละหน้าจนครบทุกหน้า (Page-by-Page Processing)
    """
    doc = doc_result["doc"]
    saved_images_meta = doc_result["saved_images"]
    
    # map metadata รูป เพื่อให้ค้นหาจาก bbox/page ได้ง่าย
    img_lookup = {}
    for img in saved_images_meta:
        # bbox จาก docling เป็น (l, t, r, b)
        bbox = img["bbox"]
        if bbox:
            # key: (page_no, bbox_top, bbox_left) -> value: image_info
            # round เพื่อแก้ปัญหาทศนิยมไม่ตรงกันเป๊ะๆ
            key = (img["page"], round(bbox[1], 1), round(bbox[0], 1))
            img_lookup[key] = img

    image_blocks = []
    
    # ----------------------------------------------------
    # 1. เตรียมข้อมูล: รวม Text และ Image ทั้งหมดในเอกสาร
    # ----------------------------------------------------
    all_elements = []
    
    # ใส่ Text ทั้งหมด
    for item in getattr(doc, "texts", []):
        if item.prov:
            p = item.prov[0]
            all_elements.append({
                "type": "text",
                "page": p.page_no,
                "top": p.bbox.t, 
                "left": p.bbox.l,
                "content": item.text,
                "obj": item
            })

    # ใส่ Image ทั้งหมด
    for item in getattr(doc, "pictures", []):
        if item.prov:
            p = item.prov[0]
            all_elements.append({
                "type": "image",
                "page": p.page_no,
                "top": p.bbox.t,
                "left": p.bbox.l,
                "obj": item
            })

    # ----------------------------------------------------
    # 2. เรียงลำดับตาม Flow การอ่านจริง (หน้า -> บนลงล่าง)
    # ----------------------------------------------------
    # การเรียงแบบนี้ทำให้เรารู้ว่า "อะไรมาก่อนรูปภาพนี้"
    all_elements.sort(key=lambda x: (x["page"], x["top"]))

    # ----------------------------------------------------
    # 3. วนลูปประมวลผล (หาบริบท + สร้างคำบรรยาย)
    # ----------------------------------------------------
    current_heading = "Unknown Section"
    last_text_paragraph = "No preceding text"
    
    img_counter = 0

    print(f"[Image Enricher] Processing {len([e for e in all_elements if e['type']=='image'])} images across all pages...")

    for elem in all_elements:
        
        # --- CASE: Text ---
        # เก็บไว้เป็นบริบท (Context) สำหรับรูปภาพที่จะเจอต่อไป
        if elem["type"] == "text":
            text = elem["content"].strip()
            if not text: continue
            
            # Logic ง่ายๆ หาหัวข้อ: ถ้าสั้นและไม่มีจุดจบประโยค ให้สมมติว่าเป็นหัวข้อ
            if len(text) < 100 and not text.endswith("."):
                current_heading = text
            
            # อัปเดตย่อหน้าล่าสุด
            last_text_paragraph = text 

        # --- CASE: Image ---
        # ต้องสร้าง ImageBlock ผูกกับบริบทที่เก็บมาล่าสุด
        elif elem["type"] == "image":
            img_counter += 1
            page = elem["page"]
            
            # หาไฟล์รูปที่เซฟไว้ (Matching logic)
            matching_saved = None
            for s_img in saved_images_meta:
                 # เช็ค page และตำแหน่ง top ใกล้เคียงกัน (+/- 5 pixel)
                 s_bbox = s_img["bbox"]
                 if s_img["page"] == page and abs(s_bbox[1] - elem["top"]) < 5.0:
                     matching_saved = s_img
                     break
            
            # ถ้าหาไฟล์รูปไม่เจอ ให้ข้าม (อาจเป็นรูปเล็กๆ ที่ถูก filter ออกตอน parse)
            if not matching_saved: 
                continue

            image_path = matching_saved["path"]
            
            # [AI] ให้ Gemini เขียนคำบรรยาย (Markdown)
            # ทำทีละรูป ทุกหน้า
            print(f"   -> Generating description for Image #{img_counter} (Page {page})...")
            description_md = ""
            if model:
                description_md = generate_image_description_md(model, image_path)
                time.sleep(1) # พักนิดนึงกัน Rate limit
            
            # [Construct Block]
            block = ImageBlock(
                id=f"img_{img_counter:04d}",
                doc_id=doc_id,
                page=page,
                file_path=image_path,
                caption=description_md,  # ผลลัพธ์จาก AI
                extra={
                    "source": "docling_contextual",
                    "context": {
                        "nearest_heading": current_heading,
                        "preceding_text": last_text_paragraph,
                        "page_topic": f"Content on Page {page}" 
                    },
                    # Metadata สำคัญสำหรับ RAG Chunking
                    "chunk_metadata": (
                        f"Image Description: {description_md[:100]}... "
                        f"| Related Section: {current_heading} "
                        f"| Context: {last_text_paragraph[:50]}..."
                    )
                }
            )
            image_blocks.append(block)

    return image_blocks


def docling_to_ingested_doc(doc_result: dict, doc_id: str, file_path: str, vision_model=None) -> IngestedDocument:
    """
    แปลงผลลัพธ์ Docling เป็น Schema ของเรา
    """
    doc = doc_result["doc"]
    
    # 1. Texts
    text_blocks = []
    if hasattr(doc, "texts"):
        for i, item in enumerate(doc.texts):
            # หา page number แบบปลอดภัย
            page_no = 1
            if item.prov:
                page_no = item.prov[0].page_no
                
            text_blocks.append(TextBlock(
                id=f"txt_{i:04d}",
                doc_id=doc_id,
                page=page_no,
                content=item.text.strip(),
                extra={"source": "docling"}
            ))

    # 2. Tables
    table_blocks = []
    if hasattr(doc, "tables"):
        for i, tbl in enumerate(doc.tables):
            try:
                # พยายาม export ให้ได้ DataFrame
                try: df = tbl.export_to_dataframe(doc=doc)
                except: df = tbl.export_to_dataframe()
                
                if df.empty: continue
                
                # หา page number
                page_no = 1
                if tbl.prov: page_no = tbl.prov[0].page_no

                table_blocks.append(TableBlock(
                    id=f"tbl_{i+1:04d}",
                    doc_id=doc_id,
                    page=page_no,
                    name=f"Table {i+1}",
                    category="docling_extracted",
                    columns=[str(c) for c in df.columns],
                    rows=df.astype(str).values.tolist(),
                    extra={"method": "docling"}
                ))
            except Exception as e:
                print(f"[Warn] Skip table {i}: {e}")

    # 3. Images (เรียกใช้ฟังก์ชันใหม่ที่วนลูปทุกหน้า)
    print(f"[run_ingestion] Enriching images with context (All Pages)...")
    image_blocks = enrich_images_with_context(doc_result, doc_id, vision_model)
    
    # คำนวณจำนวนหน้า
    page_count = 1
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

    # Init Vision Model
    # [FIX] เรียกใช้ฟังก์ชันชื่อที่ถูกต้อง
    vision_model = _get_gemini_vision_client()
    
    # 1. USE DOCLING
    print(f"==== [1/3] Ingestion (Docling) ====")
    image_dir = Path(output_root) / doc_id / "images"
    
    parser = DoclingParser(output_dir=str(Path(output_root)/doc_id), image_dir=str(image_dir))
    doc_result = parser.parse_file(str(pdf_path))
    
    # Save Markdown
    md_content = doc_result.get("markdown", "")
    if md_content:
        md_path = Path(output_root) / doc_id / f"{doc_id}.md"
        with open(md_path, "w", encoding="utf-8") as f:
            f.write(md_content)

    # 2. Convert & Enrich Images
    doc = docling_to_ingested_doc(doc_result, doc_id, str(pdf_path), vision_model=vision_model)
    doc.metadata.doc_type = doc_type

    print(f"[Docling] Extracted: Texts={len(doc.texts)}, Tables={len(doc.tables)}, Images={len(doc.images)}")
    save_ingested_document(doc, output_root)

    # 3. ENRICHMENT (Text/Table)
    print(f"==== [2/3] Enrichment (Text/Table) ====")
    try:
        predicted_type = classify_document(doc, use_gemini=True)
        doc.metadata.doc_type = predicted_type
    except: pass

    doc = tag_sections(doc, use_gemini=True)
    doc = categorize_text_blocks(doc, use_gemini=True)
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
    parser = argparse.ArgumentParser()
    parser.add_argument("pdf_path", help="Path to PDF file")
    parser.add_argument("--doc-type", default="generic")
    parser.add_argument("--doc-id", default=None)
    parser.add_argument("--output-root", default="ingested")
    args = parser.parse_args()

    run_ingestion_pipeline(
        pdf_path=args.pdf_path,
        doc_type=args.doc_type,
        doc_id=args.doc_id,
        output_root=args.output_root,
    )

if __name__ == "__main__":
    main()