from __future__ import annotations

from datetime import datetime
from typing import Any, List, Optional, Tuple, Dict, Union

from pydantic import BaseModel, Field, ConfigDict


# --------------------------------------------------------
# Basic Types
# --------------------------------------------------------

# [NEW] BBox type alias for clarity (x0, y0, x1, y1)
BBox = Tuple[float, float, float, float]


# --------------------------------------------------------
# Text / Table / Image items (มาจาก text.json / table.json / image.json)
# --------------------------------------------------------


class TextItem(BaseModel):
    """
    แทน 1 block ใน text.json ที่ฝั่ง Peng generate มา
    รองรับทั้งแบบ Normal text และ Rich metadata จาก pdf_parser
    """
    model_config = ConfigDict(extra="allow") 

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
    
    # [FIX] ย้าย extra มาไว้ที่นี่ เพื่อให้ TextItem รองรับ metadata ได้โดยตรง
    extra: Dict[str, Any] = Field(default_factory=dict)


# [CRITICAL FIX] Map TextBlock to TextItem
# เปลี่ยน TextBlock ให้เป็น Alias ของ TextItem (คือ class เดียวกัน)
# เพื่อให้ DocumentBundle ยอมรับ TextItem ที่โหลดมาจาก JSON ได้โดยไม่ติด Type Error
TextBlock = TextItem


class TableItem(BaseModel):
    """
    แทน 1 table ใน table.json
    """
    model_config = ConfigDict(extra="allow")

    id: str
    doc_id: str
    page: int

    name: Optional[str] = None
    section: Optional[str] = None
    category: Optional[str] = None

    columns: List[str] = Field(default_factory=list) # [MODIFIED] Added default factory safety
    rows: List[List[Any]] = Field(default_factory=list) # [MODIFIED] Allow Any content, default list

    bbox: Any | None = None
    doc_type: Optional[str] = None
    
    # [NEW] Support extra metadata field (like chunking.py expects)
    extra: Dict[str, Any] = Field(default_factory=dict)


class ImageItem(BaseModel):
    """
    แทน 1 รูปใน image.json
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
    
    # [NEW] Support extra metadata field
    extra: Dict[str, Any] = Field(default_factory=dict)


# --------------------------------------------------------
# Metadata (metadata.json)
# --------------------------------------------------------


class Metadata(BaseModel):
    """
    metadata.json
    """
    model_config = ConfigDict(extra="allow")

    doc_id: str
    file_name: str

    # บาง future case Peng อาจยังไม่ได้ classify doc_type → เผื่อเป็น Optional
    doc_type: Optional[str] = None

    page_count: int

    # Pydantic จะ parse string ISO8601 ให้เป็น datetime ให้เอง
    # [MODIFIED] Allow string input for flexibility (some parts might pass ISO string)
    ingested_at: Union[datetime, str]
    source: str
    
    # [NEW] Support extra metadata
    extra: Dict[str, Any] = Field(default_factory=dict)


# [NEW] Alias for pdf_parser compatibility
DocumentMetadata = Metadata


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
    # [FIX] เปลี่ยน Type เป็น TextItem เพื่อให้ตรงกับข้อมูลจริงที่เข้ามา (TextBlock คือ alias แล้ว)
    texts: List[TextItem] = Field(default_factory=list) 
    tables: List[TableItem] = Field(default_factory=list)
    images: List[ImageItem] = Field(default_factory=list)

# [NEW] Alias for pdf_parser compatibility
IngestedDocument = DocumentBundle