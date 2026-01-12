# backend/scripts/ingest_doc.py

from __future__ import annotations

import sys
import os
from pathlib import Path
from typing import Union

# เพิ่ม path ให้สามารถเรียก import folder 'scripts' ที่อยู่ root ได้
sys.path.append(os.getcwd())

from backend.services.loader import load_document_bundle
from backend.services.chunking import (
    image_items_to_chunks,
    table_items_to_chunks,
    text_items_to_chunks,
)
from backend.services.vector_store import index_chunks, search_similar

# พยายาม import pipeline หลัก
try:
    from scripts.run_ingestion import run_ingestion_pipeline
except ImportError:
    print("[WARN] ไม่สามารถ import scripts.run_ingestion ได้ (อาจต้องเช็ค path)")
    run_ingestion_pipeline = None


# -------------------------------------------------------------------
# [NEW] Function ที่ main.py เรียกใช้ (แก้ไขให้รับ output_root)
# -------------------------------------------------------------------
def run_ingestion(
    pdf_path: str, 
    doc_id: str, 
    doc_type: str = "generic_doc", 
    output_root: Union[str, Path] = "ingested"  # รับค่า Path มา
):
    """
    ฟังก์ชันหลักสำหรับรับคำสั่งจาก API (main.py)
    1. เรียก Pipeline เพื่อแปลง PDF -> JSON (Ingested)
    2. เรียก Indexing เพื่อนำ JSON -> Vector DB
    """
    print(f"[run_ingestion] Processing {doc_id} from {pdf_path}...")
    
    # แปลงให้แน่ใจว่าเป็น Path object
    output_root_path = Path(output_root)

    # 1. Run Pipeline (PDF -> JSONs)
    if run_ingestion_pipeline:
        run_ingestion_pipeline(
            pdf_path=pdf_path,
            doc_type=doc_type,
            doc_id=doc_id,
            output_root=output_root_path  # ส่ง Path ต่อไปให้ Pipeline
        )
    else:
        raise ImportError("ไม่พบฟังก์ชัน run_ingestion_pipeline ตรวจสอบว่ามีไฟล์ scripts/run_ingestion.py หรือไม่")

    # 2. Indexing (JSONs -> VectorDB)
    # --- [FIX] ใช้ output_root_path แทนการ hardcode ---
    base_dir = output_root_path / doc_id
    
    if not base_dir.exists():
         raise FileNotFoundError(f"Ingestion failed? ไม่พบโฟลเดอร์ {base_dir}")
         
    # โหลดข้อมูล (Bundle)
    bundle = load_document_bundle(str(base_dir), doc_id)
    
    # แปลงเป็น Chunks
    text_chunks = text_items_to_chunks(bundle)
    table_chunks = table_items_to_chunks(bundle)
    image_chunks = image_items_to_chunks(bundle)
    
    all_chunks = text_chunks + table_chunks + image_chunks
    
    # บันทึกลง ChromaDB
    if all_chunks:
        index_chunks(all_chunks)
        print(f"[run_ingestion] Indexed {len(all_chunks)} chunks for {doc_id}.")
    else:
        print(f"[WARN] No chunks found for {doc_id}.")


# -------------------------------------------------------------------
# CONFIG & Helpers (ของเดิม - ปรับให้รองรับ path ถ้าจำเป็น)
# -------------------------------------------------------------------
DOCS: list[tuple[str, str]] = []

def discover_docs_from_ingested(root: str = "ingested") -> list[tuple[str, str]]:
    base = Path(root)
    if not base.exists():
        print(f"[WARN] โฟลเดอร์ '{root}' ยังไม่มี")
        return []

    docs: list[tuple[str, str]] = []
    for child in base.iterdir():
        if child.is_dir():
            doc_id = child.name
            docs.append((doc_id, str(child)))
    return docs

def get_docs_to_ingest() -> list[tuple[str, str]]:
    if DOCS: return DOCS
    # ถ้าจะเทส manual ให้แก้ path ตรงนี้ด้วยถ้าต้องการ
    return discover_docs_from_ingested("ingested")

def check_ingested_folder(base_dir: str, doc_id: str) -> bool:
    base_path = Path(base_dir)
    if not (base_path / "metadata.json").exists(): return False
    return True 

# -------------------------------------------------------------------
# main (สำหรับรัน manual)
# -------------------------------------------------------------------
def main():
    docs_to_ingest = get_docs_to_ingest()
    if not docs_to_ingest:
        print("=== Ingestion: ไม่มีเอกสารให้ ingest ===")
        return

    all_chunks = []
    print("=== Ingestion: start ===")
    
    for doc_id, base_dir in docs_to_ingest:
        print(f"\n[DOC] {doc_id} from {base_dir}")
        if not check_ingested_folder(base_dir, doc_id): continue

        try:
            bundle = load_document_bundle(base_dir, doc_id)
        except Exception as e:
            print(f"[ERROR] skip {doc_id}: {e}")
            continue

        t = text_items_to_chunks(bundle)
        tb = table_items_to_chunks(bundle)
        im = image_items_to_chunks(bundle)
        
        doc_chunks = t + tb + im
        print(f"  total chunks: {len(doc_chunks)}")
        
        if doc_chunks:
            all_chunks.extend(doc_chunks)

    if all_chunks:
        index_chunks(all_chunks)
        print("\nIndexed all chunks into Chroma.")
    else:
        print("\n[SUMMARY] No chunks to index.")

if __name__ == "__main__":
    main()