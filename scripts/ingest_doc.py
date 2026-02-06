# backend/scripts/ingest_doc.py

from __future__ import annotations
import logging
import argparse
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, Any

# Services
from backend.services.loader import load_document_bundle
from backend.services.chunking import (
    text_items_to_chunks,
    image_items_to_chunks,
    # table_items_to_chunks, # เราจะใช้ smart_table_to_chunks ที่เขียน override ข้างล่างแทน
)
from backend.services.vector_store import index_chunks, search_similar

# Setup Logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# -------------------------------------------------------------------
# CONFIG
# -------------------------------------------------------------------
# ถ้าอยาก fix รายการ doc เอง ให้ใส่ใน DOCS
# ถ้าปล่อย [] ไว้ จะ auto-scan จากโฟลเดอร์ ingested/
DOCS: list[tuple[str, str]] = []

@dataclass
class Chunk:
    id: str
    content: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    # เพิ่ม 4 บรรทัดนี้ครับ
    doc_id: str = ""
    doc_type: str = "generic"
    source: str = "unknown"
    page: int = 1

# -------------------------------------------------------------------
# [HYBRID UPGRADE] Smart Table Chunking Logic
# -------------------------------------------------------------------
# -------------------------------------------------------------------
# [HYBRID UPGRADE] Smart Table Chunking Logic
# -------------------------------------------------------------------
def smart_table_to_chunks(tables: list, doc_id: str, doc_type: str = "generic"):
    """
    สร้าง Chunk สำหรับตารางแบบ Hybrid:
    1. ถ้าเป็น Complex Table -> เก็บ image_path ไว้ใน Metadata เพื่อให้ Frontend โชว์รูป
    2. เก็บ Markdown/Text ไว้ใน Content เพื่อให้ Vector Search หาเจอ
    """
    chunks = []
    for i, tbl in enumerate(tables):
        # 1. เตรียม Content (ใช้ Markdown เป็นหลักในการค้นหา)
        content = getattr(tbl, "markdown", "") or ""
        
        # Fallback: ถ้าไม่มี Markdown ให้เอา Header + Rows มาต่อกัน
        if not content and tbl.columns:
            # แปลงทุกอย่างเป็น string ก่อน join เพื่อกัน error
            header_str = " | ".join(map(str, tbl.columns))
            rows_str = "\n".join([" | ".join(map(str, row)) for row in tbl.rows[:5]]) 
            content = f"{header_str}\n{rows_str}"

        if not content.strip():
            continue

        # 2. ดึงค่าจาก Object (รองรับทั้ง Attribute และ Extra dict)
        image_path = getattr(tbl, "image_path", None)
        if not image_path and tbl.extra:
            image_path = tbl.extra.get("image_path")
            
        is_complex = getattr(tbl, "is_complex", False)
        if not is_complex and tbl.extra:
            is_complex = tbl.extra.get("is_complex", False)

        # 3. เตรียม Metadata
        meta = {
            "source": "table",
            "doc_id": doc_id,
            "page": tbl.page,
            "table_id": tbl.id,
            "image_path": image_path,
            "is_complex": bool(is_complex), 
            "name": tbl.name,
            "chunk_index": i
        }
        page_val = tbl.page if isinstance(tbl.page, int) else 1
        
        chunk = Chunk(
            id=f"{doc_id}_tbl_{i}",
            content=content, 
            metadata=meta,
            # เพิ่ม 4 บรรทัดนี้ต่อท้ายครับ
            doc_id=doc_id,
            doc_type=doc_type,
            source="table",
            page=page_val
        )
        chunks.append(chunk)
    return chunks

# -------------------------------------------------------------------
# Helper: ค้นหา doc ทั้งหมดจากโฟลเดอร์ ingested/
# -------------------------------------------------------------------
def discover_docs_from_ingested(root: str = "ingested") -> list[tuple[str, str]]:
    """
    สแกนหาเอกสารจากโฟลเดอร์ ingested/
    คืนค่าเป็น list ของ (doc_id, base_dir)
    """
    base = Path(root)
    if not base.exists():
        print(f"[WARN] โฟลเดอร์ '{root}' ยังไม่มี (ให้รัน ingestion ก่อน)")
        return []

    docs: list[tuple[str, str]] = []
    for child in base.iterdir():
        if child.is_dir():
            doc_id = child.name
            docs.append((doc_id, str(child)))
    return docs


def get_docs_to_ingest() -> list[tuple[str, str]]:
    """
    เลือกว่าจะใช้ DOCS แบบ fix เอง หรือ auto-discover จาก ingested/
    """
    if DOCS:
        print("[INFO] ใช้รายการ DOCS ที่กำหนดไว้ในสคริปต์")
        return DOCS

    print("[INFO] ไม่ได้กำหนด DOCS เอง -> scan จากโฟลเดอร์ 'ingested/'")
    docs = discover_docs_from_ingested("ingested")
    if not docs:
        print("[ERROR] ไม่พบเอกสารใน 'ingested/' เลย")
    return docs


# -------------------------------------------------------------------
# Helper: เช็คว่าโฟลเดอร์ ingested/<doc_id> มีไฟล์ "พอใช้ได้" ไหม
# -------------------------------------------------------------------
def check_ingested_folder(base_dir: str, doc_id: str) -> bool:
    """
    เช็คเบื้องต้นว่าโฟลเดอร์ ingested/<doc_id>/ นี้พอจะโหลดได้ไหม
    """
    base_path = Path(base_dir)

    meta_path = base_path / "metadata.json"
    if not meta_path.exists():
        print(f"[WARN] skip doc_id={doc_id}: ไม่มี metadata.json ใน {base_dir}")
        return False

    # TEXT check
    text_candidates = [
        base_path / "text_enriched.json",
        base_path / "text_clean.json",
        base_path / "text.json",
    ]
    if not any(p.exists() for p in text_candidates):
        print(f"[WARN] skip doc_id={doc_id}: ไม่พบไฟล์ text json")
        return False

    # TABLE check
    table_candidates = [
        base_path / "table_normalized.json",
        base_path / "table_clean.json",
        base_path / "table.json",
    ]
    if not any(p.exists() for p in table_candidates):
        print(f"[WARN] skip doc_id={doc_id}: ไม่พบไฟล์ table json")
        return False
    
    # IMAGE check
    if not (base_path / "image.json").exists():
         print(f"[WARN] skip doc_id={doc_id}: ไม่มี image.json")
         return False

    return True


# -------------------------------------------------------------------
# main
# -------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--reset", action="store_true", help="Reset DB before indexing")
    args = parser.parse_args()

    docs_to_ingest = get_docs_to_ingest()
    if not docs_to_ingest:
        print("=== Ingestion: ไม่มีเอกสารให้ ingest ===")
        return

    all_chunks = []
    ingested_doc_ids: list[str] = []

    print("=== Ingestion: start ===")
    for doc_id, base_dir in docs_to_ingest:
        print(f"\n[DOC] {doc_id} from {base_dir}")

        # 1) pre-check
        if not check_ingested_folder(base_dir, doc_id):
            continue

        # 2) โหลด DocumentBundle
        try:
            bundle = load_document_bundle(base_dir, doc_id)
        except Exception as e:
            print(f"[ERROR] skip doc_id={doc_id}: error loading bundle -> {e}")
            continue
             
        current_doc_type = "generic"
        if bundle.metadata and hasattr(bundle.metadata, "doc_type"):
            current_doc_type = bundle.metadata.doc_type
             
        # 3) แปลงเป็น chunks
        # A. Text
        text_chunks = text_items_to_chunks(bundle)
        
        # B. Table (ใช้ Smart Logic ใหม่ รองรับ Hybrid Image)
        # Note: load_document_bundle คืนค่าเป็น objects (TableBlock) เรียบร้อยแล้ว
        # เราจึงส่ง bundle.tables เข้าไปได้เลย
        table_chunks = smart_table_to_chunks(bundle.tables, doc_id, doc_type=current_doc_type)
        
        # C. Image
        image_chunks = image_items_to_chunks(bundle)

        doc_chunks = text_chunks + table_chunks + image_chunks

        print(f"  text chunks : {len(text_chunks)}")
        print(f"  table chunks: {len(table_chunks)}")
        print(f"  image chunks: {len(image_chunks)}")
        print(f"  total chunks: {len(doc_chunks)}")

        if doc_chunks:
            all_chunks.extend(doc_chunks)
            ingested_doc_ids.append(doc_id)
        else:
            print(f"[WARN] doc_id={doc_id} ไม่มี chunks เลย -> ข้าม")

    if not all_chunks:
        print("\n[SUMMARY] ไม่มี chunks เลย -> จบการทำงาน")
        return

    print(f"\n[SUMMARY] total chunks from all docs: {len(all_chunks)}")

    # 4) index chunks ทั้งหมดเข้า Chroma
    if args.reset:
        print("⚠️ Resetting Vector DB (Simulated)...")
        # ใส่ logic reset db จริงๆ ตรงนี้ถ้าต้องการ
        
    index_chunks(all_chunks)
    print("\nIndexed all chunks into Chroma.")

    # 5) ทดลอง search เบื้องต้น
    if not ingested_doc_ids:
        return

    test_queries = []
    if "doc_001" in ingested_doc_ids:
        test_queries.append(("ยอดคงเหลือรวมสิ้นงวด", ["doc_001"]))
    else:
        test_queries.append(("ทดสอบค้นหา", [ingested_doc_ids[0]]))

    for query, doc_ids in test_queries:
        print("\n" + "=" * 60)
        print(f"Test search with query: {query!r} (doc_ids={doc_ids})")
        docs = search_similar(query=query, k=3, doc_ids=doc_ids)
        if not docs:
            print("  -> No results")
            continue
        for i, doc in enumerate(docs, start=1):
            print(f"\nResult #{i}")
            print("  content :", doc.page_content[:100] + "...")
            print("  metadata:", doc.metadata)

    print("\n=== Ingestion: done ===")


if __name__ == "__main__":
    main()