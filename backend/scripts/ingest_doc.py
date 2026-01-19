# backend/scripts/ingest_doc.py

from __future__ import annotations

import sys
import os
from pathlib import Path
from typing import Union

# --- [FIX] การตั้งค่า Path ให้แม่นยำขึ้น ---
# หา Root Directory จากตำแหน่งไฟล์นี้: backend/scripts/ingest_doc.py -> parents[2] คือ Root Project
current_file = Path(__file__).resolve()
root_dir = current_file.parents[2]

# เพิ่ม Root เข้า sys.path เป็นลำดับแรก เพื่อให้ Python หา folder 'scripts' และ 'backend' เจอแน่นอน
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

# --- Import Services ---
try:
    from backend.services.loader import load_document_bundle
    from backend.services.chunking import (
        image_items_to_chunks,
        table_items_to_chunks,
        text_items_to_chunks,
    )
    from backend.services.vector_store import index_chunks
except ImportError as e:
    print(f"[CRITICAL] System Import Error (Services): {e}")
    raise e

# --- [FIX] Import Pipeline แบบแสดง Error จริง ---
try:
    # พยายาม import scripts.run_ingestion
    from scripts.run_ingestion import run_ingestion_pipeline
    print("[ingest_doc] Successfully imported run_ingestion_pipeline")
except ImportError as e:
    # พิมพ์ Error ตัวเต็มออกมาเพื่อ debug แทนที่จะข้ามไปเฉยๆ
    print(f"\n[ERROR] ไม่สามารถ import scripts.run_ingestion ได้!")
    print(f"[DEBUG] Root Dir: {root_dir}")
    print(f"[DEBUG] Sys Path: {sys.path[:3]}...")
    print(f"[DEBUG] Exception Details: {e!r}\n")
    
    # กำหนดเป็น None เพื่อให้รู้ว่าโหลดไม่สำเร็จ
    run_ingestion_pipeline = None


# -------------------------------------------------------------------
# Function ที่ main.py เรียกใช้
# -------------------------------------------------------------------
def run_ingestion(
    pdf_path: str, 
    doc_id: str, 
    doc_type: str = "generic_doc", 
    output_root: Union[str, Path] = "ingested"
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
        try:
            run_ingestion_pipeline(
                pdf_path=pdf_path,
                doc_type=doc_type,
                doc_id=doc_id,
                output_root=output_root_path
            )
        except Exception as e:
            print(f"[ERROR] Pipeline Execution Failed: {e}")
            raise e
    else:
        # ถ้า import ไม่ผ่าน ให้แจ้ง Error ชัดเจน (API จะได้ return 500 พร้อม message)
        raise ImportError(
            "ฟังก์ชัน run_ingestion_pipeline ไม่ถูกโหลด (ตรวจสอบ Log เพื่อดูสาเหตุ ImportError ของ scripts.run_ingestion)"
        )

    # 2. Indexing (JSONs -> VectorDB)
    base_dir = output_root_path / doc_id
    
    if not base_dir.exists():
         raise FileNotFoundError(f"Ingestion failed? ไม่พบโฟลเดอร์ผลลัพธ์ที่ {base_dir}")
         
    # โหลดข้อมูล (Bundle)
    try:
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
            
    except Exception as e:
        print(f"[ERROR] Indexing Failed: {e}")
        raise e


# -------------------------------------------------------------------
# Helper สำหรับ Manual Run (คงเดิม)
# -------------------------------------------------------------------
def discover_docs_from_ingested(root: str = "ingested") -> list[tuple[str, str]]:
    base = Path(root)
    if not base.exists():
        return []
    docs = []
    for child in base.iterdir():
        if child.is_dir():
            docs.append((child.name, str(child)))
    return docs

if __name__ == "__main__":
    # Manual Test Block
    print(f"Running ingest_doc.py manually. Root: {root_dir}")
    pass