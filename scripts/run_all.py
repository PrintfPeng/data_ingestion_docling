from __future__ import annotations

"""
run_all.py

Runner ตัวเดียวสำหรับรันทั้ง pipeline:

1) Ingestion  (scripts.run_ingestion.run_ingestion_pipeline)
2) Cleaning   (scripts.run_cleaning.run_cleaning)
3) Enrich     (scripts.run_semantic_enrich.run_semantic_enrich)

วิธีใช้ตัวอย่าง:

    python -m scripts.run_all samples/statement/sample.pdf --doc-id sample --use-gemini

ถ้าไม่ได้ส่ง --doc-id จะใช้ชื่อไฟล์ (ไม่รวม .pdf) เป็น doc_id ให้
"""

import argparse
from pathlib import Path

from scripts.run_ingestion import run_ingestion_pipeline
from scripts.run_cleaning import run_cleaning
from scripts.run_semantic_enrich import run_semantic_enrich


def run_all(
    pdf_path: str | Path,
    doc_id: str | None = None,
    doc_type: str = "generic",
    output_root: str | Path = "ingested",
    use_gemini: bool = False,
) -> None:
    pdf_path = Path(pdf_path)
    if doc_id is None:
        doc_id = pdf_path.stem

    print("==== [1/3] Ingestion ====")
    run_ingestion_pipeline(
        pdf_path=pdf_path,
        doc_type=doc_type,
        doc_id=doc_id,
        output_root=output_root,
    )

    print("\n==== [2/3] Cleaning ====")
    run_cleaning(
        doc_id=doc_id,
        output_root=output_root,
    )

    print("\n==== [3/3] Semantic Enrich ====")
    run_semantic_enrich(
        doc_id=doc_id,
        output_root=output_root,
        use_gemini=use_gemini,
    )

    print("\n✅ Done: full pipeline finished.")
    print(f"   - doc_id = {doc_id}")
    print(f"   - output_root = {output_root}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run full AI data ingestion pipeline (ingest + clean + enrich).")
    parser.add_argument("pdf_path", help="Path to PDF file")
    parser.add_argument(
        "--doc-id",
        default=None,
        help="Document ID (default: stem of file name)",
    )
    parser.add_argument(
        "--doc-type",
        default="generic",
        help="Doc type hint (e.g. bank_statement, invoice, receipt)",
    )
    parser.add_argument(
        "--output-root",
        default="ingested",
        help="Root folder for outputs (default: 'ingested')",
    )
    parser.add_argument(
        "--use-gemini",
        action="store_true",
        help="Enable Gemini for section/text role tagging (if GOOGLE_API_KEY is set)",
    )
    args = parser.parse_args()

    run_all(
        pdf_path=args.pdf_path,
        doc_id=args.doc_id,
        doc_type=args.doc_type,
        output_root=args.output_root,
        use_gemini=args.use_gemini,
    )


if __name__ == "__main__":
    main()
