from __future__ import annotations

"""
run_cleaning.py

ใช้ทำ Data Cleaning หลังจาก ingestion เสร็จ:
- อ่าน ingested/{doc_id}/metadata.json, text.json, table.json
- รัน cleaner.clean_text_blocks / clean_table_blocks
- เซฟเป็น text_clean.json, table_clean.json
"""

import argparse
import json
from pathlib import Path

from ingestion.schema import DocumentMetadata, TextBlock, TableBlock, IngestedDocument
from ingestion.cleaner import clean_text_blocks, clean_table_blocks


def run_cleaning(
    doc_id: str,
    output_root: str | Path = "ingested",
) -> None:
    output_root = Path(output_root)
    doc_dir = output_root / doc_id

    meta_path = doc_dir / "metadata.json"
    text_path = doc_dir / "text.json"
    table_path = doc_dir / "table.json"

    if not meta_path.exists():
        raise FileNotFoundError(f"metadata.json not found for doc_id={doc_id}")
    if not text_path.exists():
        raise FileNotFoundError(f"text.json not found for doc_id={doc_id}")

    metadata_dict = json.loads(meta_path.read_text(encoding="utf-8"))
    text_list = json.loads(text_path.read_text(encoding="utf-8"))

    meta = DocumentMetadata(**metadata_dict)
    texts = [TextBlock(**t) for t in text_list]

    if table_path.exists():
        table_list = json.loads(table_path.read_text(encoding="utf-8"))
        tables = [TableBlock(**tb) for tb in table_list]
    else:
        tables = []

    doc = IngestedDocument(
        metadata=meta,
        texts=texts,
        tables=tables,
        images=[],
    )

    print(f"[run_cleaning] Cleaning texts for doc_id={doc_id} ...")
    cleaned_texts = clean_text_blocks(doc.texts)

    print(f"[run_cleaning] Cleaning tables for doc_id={doc_id} ...")
    cleaned_tables = clean_table_blocks(doc.tables) if doc.tables else []

    text_clean_path = doc_dir / "text_clean.json"
    table_clean_path = doc_dir / "table_clean.json"

    with text_clean_path.open("w", encoding="utf-8") as f:
        json.dump(
            [t.to_dict() for t in cleaned_texts],
            f,
            ensure_ascii=False,
            indent=2,
        )

    with table_clean_path.open("w", encoding="utf-8") as f:
        json.dump(
            [tb.to_dict() for tb in cleaned_tables],
            f,
            ensure_ascii=False,
            indent=2,
        )

    print(f"[run_cleaning] Saved cleaned texts to:  {text_clean_path}")
    print(f"[run_cleaning] Saved cleaned tables to: {table_clean_path}")
    print(
        f"[run_cleaning] Done. Cleaned Texts={len(cleaned_texts)}, "
        f"Cleaned Tables={len(cleaned_tables)}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Run data cleaning for ingested document.")
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
    args = parser.parse_args()

    run_cleaning(doc_id=args.doc_id, output_root=args.output_root)


if __name__ == "__main__":
    main()
