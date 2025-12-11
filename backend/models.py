from __future__ import annotations

from datetime import datetime
from typing import Any, List, Optional

from pydantic import BaseModel, Field, ConfigDict


# --------------------------------------------------------
# Text / Table / Image items (มาจาก text.json / table.json / image.json)
# --------------------------------------------------------


class TextItem(BaseModel):
    """
    แทน 1 block ใน text.json ที่ฝั่ง Peng generate มา
    ตัวอย่าง (ฝั่ง Peng จะมี field มากกว่านี้ได้):
    {
      "id": "txt_0001",
      "doc_id": "doc_001",
      "page": 1,
      "section": "summary",
      "content": "...",
      "bbox": {...},
      "category": "body"
    }
    """

    model_config = ConfigDict(extra="allow")  # เผื่อ Peng ใส่ field อื่นเพิ่ม

    id: str
    doc_id: str
    page: int

    section: Optional[str] = None
    content: str

    # bbox จากฝั่ง Peng เป็น dict หรือ list ก็ได้
    bbox: Any | None = None

    category: Optional[str] = None

    # ฝั่ง Peng ไม่ได้ใส่ doc_type ใน text.json → ทำเป็น optional แล้วให้ loader เติมให้
    doc_type: Optional[str] = None


class TableItem(BaseModel):
    """
    แทน 1 table ใน table.json
    {
      "id": "tbl_0001",
      "doc_id": "doc_001",
      "page": 2,
      "name": "transaction_table",
      "section": "detail",
      "category": "statement_table",
      "columns": ["date", "description", "amount", "balance"],
      "rows": [
        ["2025-11-01", "โอนออก xxxxxx", "-1000.00", "5000.00"]
      ],
      "bbox": {...}
    }
    """

    model_config = ConfigDict(extra="allow")

    id: str
    doc_id: str
    page: int

    name: Optional[str] = None
    section: Optional[str] = None
    category: Optional[str] = None

    columns: List[str]
    rows: List[List[str]]

    bbox: Any | None = None
    doc_type: Optional[str] = None


class ImageItem(BaseModel):
    """
    แทน 1 รูปใน image.json
    {
      "id": "img_0001",
      "doc_id": "doc_001",
      "page": 1,
      "file_path": "images/doc_001/img_0001.png",
      "caption": "Company logo",
      "section": "header",
      "category": "logo",
      "bbox": {...}
    }
    """

    model_config = ConfigDict(extra="allow")

    id: str
    doc_id: str
    page: int

    file_path: str
    caption: Optional[str] = None

    section: Optional[str] = None
    category: Optional[str] = None

    bbox: Any | None = None
    doc_type: Optional[str] = None


# --------------------------------------------------------
# Metadata (metadata.json)
# --------------------------------------------------------


class Metadata(BaseModel):
    """
    metadata.json

    {
      "doc_id": "doc_001",
      "file_name": "my_document.pdf",
      "doc_type": "generic_doc",
      "page_count": 8,
      "ingested_at": "2025-12-01T10:00:00",
      "source": "uploaded_by_user"
    }
    """

    model_config = ConfigDict(extra="allow")

    doc_id: str
    file_name: str

    # บาง future case Peng อาจยังไม่ได้ classify doc_type → เผื่อเป็น Optional
    doc_type: Optional[str] = None

    page_count: int

    # Pydantic จะ parse string ISO8601 ให้เป็น datetime ให้เอง
    ingested_at: datetime
    source: str


# --------------------------------------------------------
# DocumentBundle – object กลางสำหรับฝั่ง RAG
# --------------------------------------------------------


class DocumentBundle(BaseModel):
    """
    รวมทุกอย่างของ doc_id เดียวกัน:
    - metadata
    - texts
    - tables
    - images

    ใช้เป็น input ให้ฟังก์ชัน chunking / embeddings / RAG
    """

    metadata: Metadata
    texts: List[TextItem] = Field(default_factory=list)
    tables: List[TableItem] = Field(default_factory=list)
    images: List[ImageItem] = Field(default_factory=list)
