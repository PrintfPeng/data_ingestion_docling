from __future__ import annotations

import json
from pathlib import Path
from typing import List

from ..models import (
    DocumentBundle,
    ImageItem,
    Metadata,
    TableItem,
    TextItem,
)


def _load_json(path: Path):
    """
    helper เล็ก ๆ โหลด JSON จากไฟล์ (ถ้าไฟล์ไม่มีจะ raise error)
    """
    if not path.exists():
        raise FileNotFoundError(f"JSON file not found: {path}")
    text = path.read_text(encoding="utf-8")
    return json.loads(text)


def _load_json_if_exists(path: Path):
    """
    helper สำหรับโหลด JSON แบบ optional
    - ถ้าไฟล์มี → คืนค่า JSON
    - ถ้าไม่มี → คืน None
    """
    if not path.exists():
        return None
    text = path.read_text(encoding="utf-8")
    return json.loads(text)


def load_document_bundle(base_dir: str, doc_id: str) -> DocumentBundle:
    """
    โหลดข้อมูลของเอกสาร 1 ชุด (doc_id เดียว) จากโฟลเดอร์ที่มีไฟล์จากฝั่ง Peng

    โครงสร้างที่คาดหวัง (จาก scripts.run_all):

        ingested/<doc_id>/
          metadata.json
          text.json
          table.json
          image.json
          text_clean.json (optional)
          table_clean.json (optional)
          text_enriched.json (optional)
          table_normalized.json (optional)
          mapping.json (optional)

    เราจะเลือกใช้ไฟล์ตาม priority:
    - Text: text_enriched.json > text_clean.json > text.json
    - Table: table_normalized.json > table_clean.json > table.json
    """

    base_path = Path(base_dir)

    # 1) metadata.json – เป็น object เดียว
    metadata_raw = _load_json(base_path / "metadata.json")
    metadata = Metadata(**metadata_raw)

    # ถ้า metadata.doc_id ไม่ตรงกับ doc_id ที่ส่งมา → เตือนเฉย ๆ แล้วใช้ของ metadata
    if metadata.doc_id != doc_id:
        print(
            f"[WARN] metadata.doc_id ({metadata.doc_id}) != requested doc_id ({doc_id}) "
            f"→ ใช้ metadata.doc_id แทน"
        )
        doc_id = metadata.doc_id

    # ----------------------------------------------------
    # 2) เลือก source สำหรับ TEXT
    # ----------------------------------------------------
    text_enriched_path = base_path / "text_enriched.json"
    text_clean_path = base_path / "text_clean.json"
    text_raw_path = base_path / "text.json"

    text_list_raw = None
    text_source_name = None

    if text_enriched_path.exists():
        text_list_raw = _load_json(text_enriched_path)
        text_source_name = "text_enriched.json"
    elif text_clean_path.exists():
        text_list_raw = _load_json(text_clean_path)
        text_source_name = "text_clean.json"
    else:
        # ถ้าไม่มี enriched/clean → fallback เป็น text.json (ต้องมีอย่างน้อยไฟล์นี้)
        text_list_raw = _load_json(text_raw_path)
        text_source_name = "text.json"

    print(f"[loader] Using {text_source_name} for doc_id={doc_id}")

    # เติม doc_id / doc_type จาก metadata (เผื่อฝั่ง Peng ไม่ได้ตั้งมาใน block)
    for item in text_list_raw:
        item.setdefault("doc_id", metadata.doc_id)
        item.setdefault("doc_type", metadata.doc_type)

    texts: List[TextItem] = [TextItem(**item) for item in text_list_raw]

    # ----------------------------------------------------
    # 3) เลือก source สำหรับ TABLE
    # ----------------------------------------------------
    table_norm_path = base_path / "table_normalized.json"
    table_clean_path = base_path / "table_clean.json"
    table_raw_path = base_path / "table.json"

    table_list_raw = None
    table_source_name = None

    if table_norm_path.exists():
        table_list_raw = _load_json(table_norm_path)
        table_source_name = "table_normalized.json"
    elif table_clean_path.exists():
        table_list_raw = _load_json(table_clean_path)
        table_source_name = "table_clean.json"
    else:
        table_list_raw = _load_json(table_raw_path)
        table_source_name = "table.json"

    print(f"[loader] Using {table_source_name} for doc_id={doc_id}")

    for item in table_list_raw:
        item.setdefault("doc_id", metadata.doc_id)
        item.setdefault("doc_type", metadata.doc_type)

    tables: List[TableItem] = [TableItem(**item) for item in table_list_raw]

    # ----------------------------------------------------
    # 4) IMAGE – ตอนนี้ใช้ image.json อย่างเดียว
    # ----------------------------------------------------
    image_raw_path = base_path / "image.json"
    image_list_raw = _load_json(image_raw_path)

    for item in image_list_raw:
        item.setdefault("doc_id", metadata.doc_id)
        item.setdefault("doc_type", metadata.doc_type)

    images: List[ImageItem] = [ImageItem(**item) for item in image_list_raw]

    # ----------------------------------------------------
    # 5) รวมทั้งหมดเป็น DocumentBundle
    # ----------------------------------------------------
    bundle = DocumentBundle(
        metadata=metadata,
        texts=texts,
        tables=tables,
        images=images,
    )
    return bundle
