from __future__ import annotations

from dataclasses import dataclass, asdict, field
from typing import List, Dict, Any, Optional, Tuple, Type, TypeVar, Union

# Bounding box: (x1, y1, x2, y2) ในหน่วยพิกัดของหน้า PDF
BBox = Tuple[float, float, float, float]


# ==========================
# DocumentMetadata
# ==========================

@dataclass
class DocumentMetadata:
    """ข้อมูลเมตาของเอกสารต้นฉบับ 1 ไฟล์"""

    doc_id: str                  # ไอดีภายในระบบ เช่น "doc_001"
    file_name: str               # ชื่อไฟล์จริง เช่น "statement_nov_2025.pdf"
    doc_type: str                # ประเภทเอกสาร เช่น "bank_statement", "receipt", "invoice"
    page_count: int              # จำนวนหน้า
    ingested_at: str             # เวลา ingest (ISO string) เช่น "2025-12-01T10:00:00"
    source: str = "uploaded"     # แหล่งที่มา เช่น "uploaded", "api", "scanner"

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls: Type["DocumentMetadata"], data: Dict[str, Any]) -> "DocumentMetadata":
        """
        โหลดจาก dict แบบ tolerant (กัน key เกิน / ขาดนิดหน่อย)
        """
        d = dict(data or {})
        return cls(
            doc_id=d.get("doc_id", ""),
            file_name=d.get("file_name", ""),
            doc_type=d.get("doc_type", "generic"),
            page_count=int(d.get("page_count", 0) or 0),
            ingested_at=d.get("ingested_at", ""),
            source=d.get("source", "uploaded"),
        )


# ==========================
# TextBlock
# ==========================

@dataclass
class TextBlock:
    """บล็อกข้อความ 1 ก้อน ในเอกสาร"""

    id: str                      # ไอดีของ block เช่น "txt_001"
    doc_id: str                  # อ้างอิงไปที่ DocumentMetadata.doc_id
    page: int                    # หน้า (เริ่มจาก 1)
    content: str                 # เนื้อความจริง ๆ
    section: Optional[str] = None    # เช่น "summary", "header", "transaction_detail"
    category: Optional[str] = None   # label ที่ใช้กับ RAG เช่น "narrative", "note"
    bbox: Optional[BBox] = None      # พิกัดบนหน้า PDF
    extra: Dict[str, Any] = field(default_factory=dict)  # ช่องไว้เก็บอะไรเพิ่มในอนาคต

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls: Type["TextBlock"], data: Dict[str, Any]) -> "TextBlock":
        """
        แปลง dict -> TextBlock แบบแข็งแรง:
        - รองรับ bbox เป็น list/tuple/None
        - extra ถ้าไม่ใช่ dict จะโดนรีเซ็ตเป็น {}
        """
        d = dict(data or {})
        bbox_raw = d.get("bbox")
        bbox: Optional[BBox] = None
        if isinstance(bbox_raw, (list, tuple)) and len(bbox_raw) == 4:
            try:
                bbox = (
                    float(bbox_raw[0]),
                    float(bbox_raw[1]),
                    float(bbox_raw[2]),
                    float(bbox_raw[3]),
                )
            except Exception:
                bbox = None

        extra = d.get("extra") or {}
        if not isinstance(extra, dict):
            extra = {}

        return cls(
            id=str(d.get("id", "")),
            doc_id=str(d.get("doc_id", "")),
            page=int(d.get("page", 1) or 1),
            content=str(d.get("content", "")),
            section=d.get("section"),
            category=d.get("category"),
            bbox=bbox,
            extra=extra,
        )


# ==========================
# TableBlock
# ==========================

@dataclass
class TableBlock:
    """โครงสร้างตาราง 1 ตาราง"""

    id: str
    doc_id: str
    page: int
    name: Optional[str] = None       # ชื่อสั้น ๆ ของตาราง เช่น "transaction_table"
    section: Optional[str] = None    # เช่น "transaction", "summary"
    category: Optional[str] = None   # เช่น "transaction_table", "item_list"
    # NOTE: ใช้ columns เป็น field หลัก แต่มี property header เป็น alias
    columns: List[str] = field(default_factory=list)
    rows: List[List[Any]] = field(default_factory=list)
    bbox: Optional[BBox] = None
    extra: Dict[str, Any] = field(default_factory=dict)

    # ---- header alias (ให้ code เก่าที่ใช้ tb.header ทำงานร่วมกันได้) ----
    @property
    def header(self) -> List[str]:
        return self.columns

    @header.setter
    def header(self, value: List[str]) -> None:
        self.columns = list(value or [])

    def to_dict(self) -> Dict[str, Any]:
        """
        คืน dict แบบเดิมที่ระบบเคยใช้:
        - ใช้ key "columns" + "rows" เป็นหลัก
        - header alias จะถูก serialize ผ่าน columns อยู่แล้ว
        """
        return asdict(self)

    @classmethod
    def from_dict(cls: Type["TableBlock"], data: Dict[str, Any]) -> "TableBlock":
        """
        รองรับทั้งรูปแบบเก่า/ใหม่:
        - ถ้ามี header/columns: จะ merge ให้ใช้เป็น columns ภายใน
        """
        d = dict(data or {})

        raw_columns = d.get("columns")
        raw_header = d.get("header")

        if raw_header is not None and raw_columns is None:
            raw_columns = raw_header
        if raw_columns is None:
            raw_columns = []

        # normalize ให้กลายเป็น list[str]
        columns: List[str] = [str(c) for c in (raw_columns or [])]

        rows_raw = d.get("rows") or []
        rows: List[List[Any]] = []
        for r in rows_raw:
            if isinstance(r, list):
                rows.append(r)
            else:
                # กันกรณีอ่านมาเป็นอย่างอื่น เช่น tuple
                rows.append(list(r))

        bbox_raw = d.get("bbox")
        bbox: Optional[BBox] = None
        if isinstance(bbox_raw, (list, tuple)) and len(bbox_raw) == 4:
            try:
                bbox = (
                    float(bbox_raw[0]),
                    float(bbox_raw[1]),
                    float(bbox_raw[2]),
                    float(bbox_raw[3]),
                )
            except Exception:
                bbox = None

        extra = d.get("extra") or {}
        if not isinstance(extra, dict):
            extra = {}

        return cls(
            id=str(d.get("id", "")),
            doc_id=str(d.get("doc_id", "")),
            page=int(d.get("page", 1) or 1),
            name=d.get("name"),
            section=d.get("section"),
            category=d.get("category"),
            columns=columns,
            rows=rows,
            bbox=bbox,
            extra=extra,
        )


# ==========================
# ImageBlock
# ==========================

@dataclass
class ImageBlock:
    """ข้อมูลรูปภาพ 1 รูป ในเอกสาร"""

    id: str
    doc_id: str
    page: int
    file_path: str                   # path ไฟล์รูป เช่น "images/doc_001/img_001.png"
    caption: Optional[str] = None    # caption หรือข้อความรอบ ๆ รูป
    section: Optional[str] = None
    category: Optional[str] = None   # เช่น "logo", "chart", "signature"
    bbox: Optional[BBox] = None
    extra: Dict[str, Any] = field(default_factory=dict)

    # alias image_path -> file_path (กันโค้ดที่ใช้คนละชื่อ)
    @property
    def image_path(self) -> str:
        return self.file_path

    @image_path.setter
    def image_path(self, value: str) -> None:
        self.file_path = value

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls: Type["ImageBlock"], data: Dict[str, Any]) -> "ImageBlock":
        """
        รองรับทั้ง key file_path และ image_path
        """
        d = dict(data or {})

        file_path = d.get("file_path") or d.get("image_path") or ""

        bbox_raw = d.get("bbox")
        bbox: Optional[BBox] = None
        if isinstance(bbox_raw, (list, tuple)) and len(bbox_raw) == 4:
            try:
                bbox = (
                    float(bbox_raw[0]),
                    float(bbox_raw[1]),
                    float(bbox_raw[2]),
                    float(bbox_raw[3]),
                )
            except Exception:
                bbox = None

        extra = d.get("extra") or {}
        if not isinstance(extra, dict):
            extra = {}

        return cls(
            id=str(d.get("id", "")),
            doc_id=str(d.get("doc_id", "")),
            page=int(d.get("page", 1) or 1),
            file_path=str(file_path),
            caption=d.get("caption"),
            section=d.get("section"),
            category=d.get("category"),
            bbox=bbox,
            extra=extra,
        )


# ==========================
# IngestedDocument
# ==========================

TIngested = TypeVar("TIngested", bound="IngestedDocument")


@dataclass
class IngestedDocument:
    """
    ตัวแทนผลลัพธ์การ ingest เอกสาร 1 ไฟล์
    รวม metadata + text blocks + tables + images
    """

    metadata: DocumentMetadata
    texts: List[TextBlock] = field(default_factory=list)
    tables: List[TableBlock] = field(default_factory=list)
    images: List[ImageBlock] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """แปลงทั้งเอกสารเป็น dict พร้อมสำหรับ serialize เป็น JSON"""
        return {
            "metadata": self.metadata.to_dict(),
            "texts": [t.to_dict() for t in self.texts],
            "tables": [tb.to_dict() for tb in self.tables],
            "images": [im.to_dict() for im in self.images],
        }

    @classmethod
    def from_dict(cls: Type[TIngested], data: Dict[str, Any]) -> TIngested:
        """
        โหลด IngestedDocument กลับจาก dict/JSON:
        - สร้าง metadata, texts, tables, images แบบ strongly-typed
        """
        d = dict(data or {})

        meta_raw = d.get("metadata") or {}
        texts_raw = d.get("texts") or []
        tables_raw = d.get("tables") or []
        images_raw = d.get("images") or []

        metadata = DocumentMetadata.from_dict(meta_raw)
        texts = [TextBlock.from_dict(t) for t in texts_raw]
        tables = [TableBlock.from_dict(tb) for tb in tables_raw]
        images = [ImageBlock.from_dict(im) for im in images_raw]

        return cls(
            metadata=metadata,
            texts=texts,
            tables=tables,
            images=images,
        )
