# scripts/run_ingestion.py

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
from ingestion.image_extractor import _get_gemini_vision_client, generate_image_description_md


def enrich_images_with_context(doc_result: dict, doc_id: str, model) -> list[ImageBlock]:
    """
    ฟังก์ชันผูกบริบท Text รอบข้างเข้ากับรูปภาพ และให้ Gemini เขียนคำบรรยาย
    """
    doc = doc_result["doc"]
    saved_images_meta = doc_result["saved_images"]
    
    # map metadata รูป
    img_lookup = {}
    for img in saved_images_meta:
        bbox = img["bbox"]
        if bbox:
            key = (img["page"], round(bbox[1], 1), round(bbox[0], 1))
            img_lookup[key] = img

    image_blocks = []
    
    # 1. เตรียมข้อมูล Text และ Image
    all_elements = []
    for item in getattr(doc, "texts", []):
        if item.prov:
            p = item.prov[0]
            all_elements.append({
                "type": "text", "page": p.page_no, "top": p.bbox.t, "left": p.bbox.l,
                "content": item.text, "obj": item
            })

    for item in getattr(doc, "pictures", []):
        if item.prov:
            p = item.prov[0]
            all_elements.append({
                "type": "image", "page": p.page_no, "top": p.bbox.t, "left": p.bbox.l,
                "obj": item
            })

    all_elements.sort(key=lambda x: (x["page"], x["top"]))

    # 2. วนลูปประมวลผล
    current_heading = "Unknown Section"
    last_text_paragraph = "No preceding text"
    img_counter = 0

    print(f"[Image Enricher] Processing {len([e for e in all_elements if e['type']=='image'])} images...")

    for elem in all_elements:
        if elem["type"] == "text":
            text = elem["content"].strip()
            if not text: continue
            if len(text) < 100 and not text.endswith("."):
                current_heading = text
            last_text_paragraph = text 

        elif elem["type"] == "image":
            img_counter += 1
            page = elem["page"]
            
            matching_saved = None
            for s_img in saved_images_meta:
                 s_bbox = s_img["bbox"]
                 if s_img["page"] == page and abs(s_bbox[1] - elem["top"]) < 5.0:
                     matching_saved = s_img
                     break
            
            if not matching_saved: continue

            image_path = matching_saved["path"]
            
            # [AI] ให้ Gemini เขียนคำบรรยาย พร้อม Retry Logic
            print(f"   -> Generating description for Image #{img_counter} (Page {page})...")
            description_md = ""
            
            if model:
                max_retries = 3
                for attempt in range(max_retries):
                    try:
                        description_md = generate_image_description_md(model, image_path)
                        if description_md:
                            time.sleep(5)
                            break
                    except Exception as e:
                        print(f"      [Warn] AI Error: {e}")
                    
                    wait_time = 10 * (attempt + 1)
                    print(f"      [Retry {attempt+1}/{max_retries}] Waiting {wait_time}s before retrying...")
                    time.sleep(wait_time)
            
            block = ImageBlock(
                id=f"img_{img_counter:04d}",
                doc_id=doc_id,
                page=page,
                file_path=image_path,
                caption=description_md,
                extra={
                    "source": "docling_contextual",
                    "context": {
                        "nearest_heading": current_heading,
                        "preceding_text": last_text_paragraph,
                        "page_topic": f"Content on Page {page}" 
                    },
                    "chunk_metadata": (
                        f"Image Description: {description_md[:100]}... "
                        f"| Related Section: {current_heading} "
                    )
                }
            )
            image_blocks.append(block)

    return image_blocks


def docling_to_ingested_doc(doc_result: dict, doc_id: str, file_path: str, vision_model=None) -> IngestedDocument:
    doc = doc_result["doc"]
    
    text_blocks = []
    if hasattr(doc, "texts"):
        for i, item in enumerate(doc.texts):
            page_no = 1
            if item.prov: page_no = item.prov[0].page_no
            text_blocks.append(TextBlock(
                id=f"txt_{i:04d}", doc_id=doc_id, page=page_no,
                content=item.text.strip(), extra={"source": "docling"}
            ))

    table_blocks = []
    if hasattr(doc, "tables"):
        for i, tbl in enumerate(doc.tables):
            try:
                try: df = tbl.export_to_dataframe(doc=doc)
                except: df = tbl.export_to_dataframe()
                if df.empty: continue
                page_no = 1
                if tbl.prov: page_no = tbl.prov[0].page_no
                table_blocks.append(TableBlock(
                    id=f"tbl_{i+1:04d}", doc_id=doc_id, page=page_no,
                    name=f"Table {i+1}", category="docling_extracted",
                    columns=[str(c) for c in df.columns],
                    rows=df.astype(str).values.tolist(), extra={"method": "docling"}
                ))
            except Exception as e: print(f"[Warn] Skip table {i}: {e}")

    print(f"[run_ingestion] Enriching images with context (All Pages)...")
    image_blocks = enrich_images_with_context(doc_result, doc_id, vision_model)
    
    page_count = 1
    all_pages = [t.page for t in text_blocks] + [i.page for i in image_blocks]
    if all_pages: page_count = max(all_pages)

    return IngestedDocument(
        metadata=DocumentMetadata(
            doc_id=doc_id, file_name=Path(file_path).name, doc_type="generic",
            page_count=page_count, ingested_at=datetime.now().isoformat(), source="docling"
        ),
        texts=text_blocks, tables=table_blocks, images=image_blocks
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
    if not doc_id: doc_id = pdf_path.stem
    vision_model = _get_gemini_vision_client()
    
    print(f"==== [1/3] Ingestion (Docling) ====")
    image_dir = Path(output_root) / doc_id / "images"
    parser = DoclingParser(output_dir=str(Path(output_root)/doc_id), image_dir=str(image_dir))
    doc_result = parser.parse_file(str(pdf_path))
    
    # [NEW LOGIC] แทนที่ ด้วย Link รูปภาพจริงๆ
    md_content = doc_result.get("markdown", "")
    saved_images = doc_result.get("saved_images", [])

    if md_content:
        # วนลูปแทนที่ทีละรูป (Docling มักจะส่งมาเรียงตามลำดับอยู่แล้ว)
        for img in saved_images:
            img_name = Path(img["path"]).name
            # Path ที่ Frontend เข้าใจได้
            web_path = f"/ingested/{doc_id}/images/{img_name}"
            
            # แทนที่ ด้วย Markdown Link
            # replace(..., 1) เพื่อแทนที่ทีละ 1 จุด เรียงตามลำดับ
            md_content = md_content.replace("", f"![{img_name}]({web_path})", 1)

        # Save ไฟล์ .md
        md_path = Path(output_root) / doc_id / f"{doc_id}.md"
        try:
            with open(md_path, "w", encoding="utf-8") as f:
                f.write(md_content)
            print(f"[Docling] Saved enhanced Markdown to: {md_path}")
        except Exception as e:
            print(f"[Warn] Could not save markdown file: {e}")

    doc = docling_to_ingested_doc(doc_result, doc_id, str(pdf_path), vision_model=vision_model)
    doc.metadata.doc_type = doc_type
    print(f"[Docling] Extracted: Texts={len(doc.texts)}, Tables={len(doc.tables)}, Images={len(doc.images)}")
    save_ingested_document(doc, output_root)

    print(f"==== [2/3] Enrichment (Text/Table) ====")
    try:
        predicted_type = classify_document(doc, use_gemini=True)
        doc.metadata.doc_type = predicted_type
    except: pass

    doc = tag_sections(doc, use_gemini=True)
    doc = categorize_text_blocks(doc, use_gemini=True)
    doc.tables = normalize_tables(doc.tables)
    
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