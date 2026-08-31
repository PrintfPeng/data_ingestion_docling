"""
scripts/rebuild_index.py

Rebuild the entire ChromaDB from already-parsed documents in ingested/.
Useful after changing the chunking strategy (e.g. enabling contextual
chunking) — regenerates every embedding with the new pipeline.

Runs from inside the container:
  docker exec ingestion-backend python scripts/rebuild_index.py

Requires ingested/{doc_id}/ folders to already exist (parsed by Docling).
"""
from __future__ import annotations

import sys
import shutil
import logging
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.ingest_doc import (
    load_document_bundle,
    check_ingested_folder,
    smart_table_to_chunks,
)
from backend.services.chunking import (
    text_items_to_chunks,
    image_items_to_chunks,
)
from backend.services.vector_store import (
    index_chunks,
    get_vector_store,
    CHROMA_DIR,
    COLLECTION_NAME,
    reset_vector_store_cache,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("rebuild")


INGESTED_DIR = Path("ingested")


def wipe_chroma():
    """Delete the entire Chroma collection so re-ingest starts clean."""
    reset_vector_store_cache()
    # Delete the persistent dir
    p = Path(CHROMA_DIR)
    if p.exists():
        logger.warning(f"Deleting {p} ...")
        shutil.rmtree(p, ignore_errors=True)
    p.mkdir(parents=True, exist_ok=True)
    logger.info("Chroma wiped.")


def rebuild_all():
    if not INGESTED_DIR.exists():
        logger.error(f"{INGESTED_DIR} not found — nothing to rebuild")
        return

    doc_ids = [p.name for p in INGESTED_DIR.iterdir() if p.is_dir()]
    logger.info(f"Found {len(doc_ids)} ingested doc(s): {doc_ids}")
    if not doc_ids:
        return

    wipe_chroma()

    all_chunks = []
    for doc_id in doc_ids:
        logger.info(f"--- {doc_id} ---")
        # check_ingested_folder + load_document_bundle expect the DOC-LEVEL
        # folder, not the parent 'ingested/' root
        doc_dir = INGESTED_DIR / doc_id
        if not check_ingested_folder(str(doc_dir), doc_id):
            logger.warning(f"skip {doc_id} — folder incomplete")
            continue
        try:
            bundle = load_document_bundle(str(doc_dir), doc_id)
        except Exception as e:
            logger.error(f"load bundle failed for {doc_id}: {e}")
            continue

        current_doc_type = "generic"
        if bundle.metadata and hasattr(bundle.metadata, "doc_type"):
            current_doc_type = bundle.metadata.doc_type

        text_chunks = text_items_to_chunks(bundle)
        table_chunks = smart_table_to_chunks(bundle.tables, doc_id, doc_type=current_doc_type)
        image_chunks = image_items_to_chunks(bundle)
        doc_chunks = text_chunks + table_chunks + image_chunks
        logger.info(f"  chunks: text={len(text_chunks)} · table={len(table_chunks)} · image={len(image_chunks)}")
        all_chunks.extend(doc_chunks)

    logger.info(f"Total chunks: {len(all_chunks)}")
    if not all_chunks:
        logger.error("No chunks — abort")
        return

    logger.info("Indexing with contextual chunking...")
    index_chunks(all_chunks)
    logger.info("Done.")


if __name__ == "__main__":
    rebuild_all()
