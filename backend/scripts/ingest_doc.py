# backend/scripts/ingest_doc.py

from __future__ import annotations

from pathlib import Path

from backend.services.loader import load_document_bundle
from backend.services.chunking import (
    image_items_to_chunks,
    table_items_to_chunks,
    text_items_to_chunks,
)
from backend.services.vector_store import index_chunks, search_similar


# -------------------------------------------------------------------
# CONFIG
# -------------------------------------------------------------------
DOCS: list[tuple[str, str]] = []


# -------------------------------------------------------------------
# Helper: ค้นหา doc ทั้งหมดจากโฟลเดอร์ ingested/
# -------------------------------------------------------------------
def discover_docs_from_ingested(root: str = "ingested") -> list[tuple[str, str]]:
    base = Path(root)
    if not base.exists():
        print(f"[WARN] โฟลเดอร์ '{root}' ยังไม่มี (ให้ฝั่ง Peng รัน ingestion ก่อน)")
        return []

    docs: list[tuple[str, str]] = []
    for child in base.iterdir():
        if child.is_dir():
            doc_id = child.name
            docs.append((doc_id, str(child)))
    return docs


def get_docs_to_ingest() -> list[tuple[str, str]]:
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
    base_path = Path(base_dir)

    meta_path = base_path / "metadata.json"
    if not meta_path.exists():
        print(f"[WARN] skip doc_id={doc_id}: ไม่มี metadata.json ใน {base_dir}")
        return False

    text_candidates = [
        base_path / "text_enriched.json",
        base_path / "text_clean.json",
        base_path / "text.json",
    ]
    if not any(p.exists() for p in text_candidates):
        print(
            f"[WARN] skip doc_id={doc_id}: "
            f"ไม่พบ text_enriched.json / text_clean.json / text.json ใน {base_dir}"
        )
        return False

    table_candidates = [
        base_path / "table_normalized.json",
        base_path / "table_clean.json",
        base_path / "table.json",
    ]
    if not any(p.exists() for p in table_candidates):
        print(
            f"[WARN] skip doc_id={doc_id}: "
            f"ไม่พบ table_normalized.json / table_clean.json / table.json ใน {base_dir}"
        )
        return False

    image_path = base_path / "image.json"
    if not image_path.exists():
        print(f"[WARN] skip doc_id={doc_id}: ไม่มี image.json ใน {base_dir}")
        return False

    return True


# -------------------------------------------------------------------
# main
# -------------------------------------------------------------------
def main():
    docs_to_ingest = get_docs_to_ingest()
    if not docs_to_ingest:
        print("=== Ingestion: ไม่มีเอกสารให้ ingest ===")
        return

    all_chunks = []
    ingested_doc_ids: list[str] = []

    print("=== Ingestion: start ===")
    for doc_id, base_dir in docs_to_ingest:
        print(f"\n[DOC] {doc_id} from {base_dir}")

        if not check_ingested_folder(base_dir, doc_id):
            continue

        try:
            bundle = load_document_bundle(base_dir, doc_id)
        except FileNotFoundError as e:
            print(f"[ERROR] skip doc_id={doc_id}: file not found -> {e}")
            continue
        except ValueError as e:
            print(f"[ERROR] skip doc_id={doc_id}: value error -> {e}")
            continue
        except Exception as e:
            print(f"[ERROR] skip doc_id={doc_id}: unexpected error -> {e}")
            continue

        text_chunks = text_items_to_chunks(bundle)
        table_chunks = table_items_to_chunks(bundle)
        image_chunks = image_items_to_chunks(bundle)

        doc_chunks = text_chunks + table_chunks + image_chunks

        print(f"  text chunks : {len(text_chunks)}")
        print(f"  table chunks: {len(table_chunks)}")
        print(f"  image chunks: {len(image_chunks)}")
        print(f"  total chunks: {len(doc_chunks)}")

        if doc_chunks:
            all_chunks.extend(doc_chunks)
            # --- FIX: ใช้ ID จริงจาก Metadata แทนชื่อโฟลเดอร์ ---
            real_id = bundle.metadata.doc_id
            ingested_doc_ids.append(real_id)
        else:
            print(f"[WARN] doc_id={doc_id} ไม่มี chunks เลย → ข้ามจากการ index")

    if not all_chunks:
        print("\n[SUMMARY] ไม่มี chunks จากเอกสารไหนเลย → ไม่เรียก index_chunks")
        print("=== Ingestion: done (no data) ===")
        return

    print(f"\n[SUMMARY] total chunks from all docs: {len(all_chunks)}")

    index_chunks(all_chunks)
    print("\nIndexed all chunks into Chroma.")

    if not ingested_doc_ids:
        print("\n[INFO] ไม่มี doc ไหน ingest สำเร็จ → ข้าม test search")
        print("\n=== Ingestion: done ===")
        return

    # Test Search Logic
    test_queries: list[tuple[str, list[str]]] = []

    # ลองสุ่มหาจาก doc ตัวแรกที่ ingest สำเร็จ
    first_doc = ingested_doc_ids[0]
    test_queries.append(("ยอดคงเหลือรวมสิ้นงวด", [first_doc]))

    for query, doc_ids in test_queries:
        print("\n" + "=" * 60)
        print(f"Test search with query: {query!r} (doc_ids={doc_ids})")

        docs = search_similar(query=query, k=3, doc_ids=doc_ids)

        if not docs:
            print("  -> No results")
            continue

        for i, doc in enumerate(docs, start=1):
            print(f"\nResult #{i}")
            print("  content :", doc.page_content[:200] + "...")
            print("  metadata:", doc.metadata)

    print("\n=== Ingestion: done ===")


if __name__ == "__main__":
    main()