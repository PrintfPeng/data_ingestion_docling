from __future__ import annotations

"""
run_semantic_enrich.py

ขั้นตอนหลัง ingestion + cleaning:
1) โหลด metadata + text_clean + table_clean
2) ใส่ section label ให้ text (tag_sections)
3) normalize header ของ table (normalize_tables)
4) extract transaction records (prepare_mapping_payload)
5) เซฟออกเป็น:
    - text_enriched.json
    - table_normalized.json
    - mapping.json
"""

import argparse
import json
from pathlib import Path

from ingestion.schema import DocumentMetadata, TextBlock, TableBlock, IngestedDocument
from ingestion.semantic_enricher import (
    tag_sections,
    normalize_tables,
    prepare_mapping_payload,
)


def run_semantic_enrich(
    doc_id: str,
    output_root: str | Path = "ingested",
    use_gemini: bool = False,
) -> None:
    output_root = Path(output_root)
    doc_dir = output_root / doc_id

    meta_path = doc_dir / "metadata.json"
    text_clean_path = doc_dir / "text_clean.json"
    table_clean_path = doc_dir / "table_clean.json"

    if not meta_path.exists():
        raise FileNotFoundError(f"metadata.json not found for doc_id={doc_id}")
    if not text_clean_path.exists():
        raise FileNotFoundError(f"text_clean.json not found for doc_id={doc_id}. Please run cleaning first.")

    metadata_dict = json.loads(meta_path.read_text(encoding="utf-8"))
    text_list = json.loads(text_clean_path.read_text(encoding="utf-8"))
    meta = DocumentMetadata(**metadata_dict)
    texts = [TextBlock(**t) for t in text_list]

    if table_clean_path.exists():
        table_list = json.loads(table_clean_path.read_text(encoding="utf-8"))
        tables = [TableBlock(**tb) for tb in table_list]
    else:
        tables = []

    doc = IngestedDocument(
        metadata=meta,
        texts=texts,
        tables=tables,
        images=[],
    )

    # 1) tag sections in text
    print(f"[run_semantic_enrich] Tagging sections (use_gemini={use_gemini}) ...")
    doc = tag_sections(doc, use_gemini=use_gemini)

    # 2) normalize tables
    print("[run_semantic_enrich] Normalizing tables ...")
    doc.tables = normalize_tables(doc.tables)

    # 3) prepare mapping payload
    print("[run_semantic_enrich] Preparing mapping payload ...")
    mapping = prepare_mapping_payload(doc)

    # save outputs
    text_enriched_path = doc_dir / "text_enriched.json"
    table_normalized_path = doc_dir / "table_normalized.json"
    mapping_path = doc_dir / "mapping.json"

    with text_enriched_path.open("w", encoding="utf-8") as f:
        json.dump(
            [t.to_dict() for t in doc.texts],
            f,
            ensure_ascii=False,
            indent=2,
        )

    with table_normalized_path.open("w", encoding="utf-8") as f:
        json.dump(
            [tb.to_dict() for tb in doc.tables],
            f,
            ensure_ascii=False,
            indent=2,
        )

    with mapping_path.open("w", encoding="utf-8") as f:
        json.dump(
            mapping,
            f,
            ensure_ascii=False,
            indent=2,
        )

    print(f"[run_semantic_enrich] Saved text_enriched to:   {text_enriched_path}")
    print(f"[run_semantic_enrich] Saved table_normalized to: {table_normalized_path}")
    print(f"[run_semantic_enrich] Saved mapping to:          {mapping_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run semantic enrichment pipeline (section + normalize + mapping).")
    parser.add_argument(
        "--doc-id",
        required=True,
        help="Document ID (เช่น 'sample')",
    )
    parser.add_argument(
        "--output-root",
        default="ingested",
        help="Root folder ของผล ingestion (default: 'ingested')",
    )
    parser.add_argument(
        "--use-gemini",
        action="store_true",
        help="ถ้าระบุ flag นี้ จะให้ Gemini ช่วย tag section",
    )
    args = parser.parse_args()

    run_semantic_enrich(
        doc_id=args.doc_id,
        output_root=args.output_root,
        use_gemini=args.use_gemini,
    )


if __name__ == "__main__":
    main()
