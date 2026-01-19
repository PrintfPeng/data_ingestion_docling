from __future__ import annotations

from dataclasses import dataclass, asdict, field
from typing import List, Dict, Any, Optional, Tuple, Type, TypeVar, Union

# =============================================================================
# GLOBAL TYPES & HELPERS
# =============================================================================

# Bounding box: (x1, y1, x2, y2) in PDF page coordinates
BBox = Tuple[float, float, float, float]

def _safe_bbox(bbox_raw: Any) -> Optional[BBox]:
    """
    Helper to safely parse a bounding box from arbitrary input.
    Returns (x1, y1, x2, y2) as floats, or None if invalid.
    """
    if isinstance(bbox_raw, (list, tuple)) and len(bbox_raw) == 4:
        try:
            return (
                float(bbox_raw[0]),
                float(bbox_raw[1]),
                float(bbox_raw[2]),
                float(bbox_raw[3]),
            )
        except (ValueError, TypeError):
            return None
    return None

def _safe_list(value: Any) -> List[Any]:
    """Helper to ensure value is a list."""
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    return []

def _safe_dict(value: Any) -> Dict[str, Any]:
    """Helper to ensure value is a dict."""
    if isinstance(value, dict):
        return value
    return {}

def _normalize_str(value: Any) -> Optional[str]:
    """
    Normalizes semantic strings: lowercase, stripped.
    Returns None if empty or non-string.
    """
    if isinstance(value, str):
        s = value.strip().lower()
        if s:
            return s
    return None

def _normalize_enum(value: Any, valid_set: set[str], default: str) -> str:
    """
    Normalizes a string against a known set of valid values.
    Returns default if unknown.
    """
    norm = _normalize_str(value)
    if norm in valid_set:
        return norm
    return default

# =============================================================================
# DocumentMetadata
# =============================================================================

@dataclass
class DocumentMetadata:
    """
    Metadata for a single source document.
    """
    doc_id: str
    file_name: str
    doc_type: str                  # e.g., "bank_statement", "invoice"
    page_count: int
    ingested_at: str               # ISO string, e.g., "2025-12-01T10:00:00"
    source: str = "uploaded"       # e.g., "uploaded", "api", "scanner"

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls: Type["DocumentMetadata"], data: Dict[str, Any]) -> "DocumentMetadata":
        d = _safe_dict(data)
        return cls(
            doc_id=str(d.get("doc_id", "")),
            file_name=str(d.get("file_name", "")),
            doc_type=_normalize_str(d.get("doc_type")) or "generic",
            page_count=int(d.get("page_count", 0) or 0),
            ingested_at=str(d.get("ingested_at", "")),
            source=_normalize_str(d.get("source")) or "uploaded",
        )


# =============================================================================
# TextBlock
# =============================================================================

@dataclass
class TextBlock:
    """
    Represents a specific block of text within the document.
    """
    id: str
    doc_id: str
    page: int
    content: str
    section: Optional[str] = None      # e.g., "header", "footer", "body"
    category: Optional[str] = None     # RAG label, e.g., "narrative", "legal_clause"
    role: Optional[str] = None         # e.g., "title", "paragraph"
    bbox: Optional[BBox] = None
    extra: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls: Type["TextBlock"], data: Dict[str, Any]) -> "TextBlock":
        d = _safe_dict(data)
        return cls(
            id=str(d.get("id", "")),
            doc_id=str(d.get("doc_id", "")),
            page=int(d.get("page", 1) or 1),
            content=str(d.get("content", "")),
            section=_normalize_str(d.get("section")),
            category=_normalize_str(d.get("category")),
            role=_normalize_str(d.get("role")),
            bbox=_safe_bbox(d.get("bbox")),
            extra=_safe_dict(d.get("extra")),
        )


# =============================================================================
# TableBlock
# =============================================================================

@dataclass
class TableBlock:
    """
    Represents a table extracted from the document.
    Supports both high-fidelity structural extraction (Camelot) and 
    lossy/vision-based extraction (GPT-4o/OCR).
    """

    # --- Identity & Location ---
    id: str
    doc_id: str
    page: int
    
    # --- Semantic Metadata ---
    name: Optional[str] = None         # e.g., "Balance Sheet"
    section: Optional[str] = None      # e.g., "financials"
    category: Optional[str] = None     # e.g., "financial_table"
    role: Optional[str] = None         # e.g., "data", "layout", "reference"

    # --- Content (Structured) ---
    # Primary data storage. 'columns' acts as the header.
    columns: List[str] = field(default_factory=list)
    rows: List[List[Any]] = field(default_factory=list)

    # --- Content (Representations) ---
    markdown: Optional[str] = None     # LLM-ready markdown representation
    html_content: Optional[str] = None # HTML representation for UI rendering

    # --- Extraction Metadata (Future Proofing) ---
    source: str = "unknown"            # e.g., "camelot", "vision", "layout_model"
    method: Optional[str] = None       # Specific algorithm e.g., "lattice", "stream", "gpt4v"
    numeric_trust: str = "unknown"     # "high" (programmatic), "medium", "low" (vision/generative)
    
    # --- Flags ---
    structured_available: bool = False # True if columns/rows are reliable
    raw_available: bool = False        # True if raw text content is preserved
    structure_lossy: bool = False      # True if structure might be hallucinatory (Vision)

    # --- Geometry & Extensions ---
    bbox: Optional[BBox] = None
    extra: Dict[str, Any] = field(default_factory=dict)
    """
    Extra container for arbitrary metadata.
    Recommended keys:
      - extra["extraction"]: Dict details about the engine configuration.
      - extra["confidence"]: float (0.0-1.0).
      - extra["model"]: str name of the AI model used.
      - extra["summary"]: str summary of the table context.
    """

    # --- Backward Compatibility: Header Alias ---
    @property
    def header(self) -> List[str]:
        """Alias for columns to maintain backward compatibility."""
        return self.columns

    @header.setter
    def header(self, value: List[str]) -> None:
        self.columns = list(value or [])

    def to_dict(self) -> Dict[str, Any]:
        """
        Serializes to dictionary. 
        Note: Properties (like header) are not included by asdict, 
        but 'columns' is included, which preserves the data.
        """
        return asdict(self)

    @classmethod
    def from_dict(cls: Type["TableBlock"], data: Dict[str, Any]) -> "TableBlock":
        """
        Robust loader supporting legacy data and new fields.
        Priority: columns > header
        """
        d = _safe_dict(data)

        # >>> SCHEMA TOLERANCE FIX <<<
        # 1. Harvest 'extra' data: Start with explicit extra, then fold in unknown root keys
        # This ensures fields like 'summary', 'html', 'markdown_content' from extractors 
        # are preserved in extra if not explicitly mapped.
        extra_data = _safe_dict(d.get("extra")).copy()
        
        # Define known fields to exclude from extra (including properties/aliases)
        known_fields = {
            "id", "doc_id", "page", 
            "name", "section", "category", "role",
            "columns", "rows", "header",
            "markdown", "html_content", 
            "source", "method", "numeric_trust",
            "structured_available", "raw_available", "structure_lossy",
            "bbox", "extra"
        }

        # Collect unexpected fields into extra
        for k, v in d.items():
            if k not in known_fields:
                extra_data[k] = v

        # 2. Handle Header/Column Compatibility
        raw_columns = d.get("columns")
        raw_header = d.get("header")
        
        if raw_columns is None and raw_header is not None:
            raw_columns = raw_header
        
        columns: List[str] = [str(c) for c in _safe_list(raw_columns)]

        # 3. Handle Rows
        rows_raw = _safe_list(d.get("rows"))
        rows: List[List[Any]] = []
        for r in rows_raw:
            if isinstance(r, (list, tuple)):
                rows.append(list(r))
            else:
                rows.append([]) # Fail soft on invalid row structure

        # 4. Normalize Semantic Fields with Alias Support
        # >>> SCHEMA TOLERANCE FIX: Check aliases for markdown/html <<<
        
        # Markdown: Check root -> alias -> extra
        markdown_val = d.get("markdown")
        if markdown_val is None:
            markdown_val = d.get("markdown_content")
        if markdown_val is None:
            markdown_val = extra_data.get("markdown")

        # HTML: Check root -> alias -> extra
        html_val = d.get("html_content")
        if html_val is None:
            html_val = d.get("html")
        if html_val is None:
            html_val = extra_data.get("html_content")
            
        # Normalizers
        source_val = _normalize_str(d.get("source")) or "unknown"
        numeric_trust_val = _normalize_enum(
            d.get("numeric_trust"), 
            {"high", "medium", "low", "unknown"}, 
            "unknown"
        )

        # 5. Implicit Flags (Default Inference)
        structured_avail = d.get("structured_available")
        if structured_avail is None:
            # Default to True if we have actual data
            structured_avail = bool(columns and rows)
        else:
            structured_avail = bool(structured_avail)

        raw_avail = d.get("raw_available")
        if raw_avail is None:
            # Default to True if we have raw string representations
            raw_avail = bool(markdown_val or html_val)
        else:
            raw_avail = bool(raw_avail)

        lossy = d.get("structure_lossy")
        if lossy is None:
            # Infer lossiness from source or trust level
            if source_val == "vision" or numeric_trust_val == "low":
                lossy = True
            else:
                lossy = False
        else:
            lossy = bool(lossy)

        # 6. Construct Object
        return cls(
            id=str(d.get("id", "")),
            doc_id=str(d.get("doc_id", "")),
            page=int(d.get("page", 1) or 1),
            name=d.get("name"),  # Keep name case-sensitive
            section=_normalize_str(d.get("section")),
            category=_normalize_str(d.get("category")),
            role=_normalize_str(d.get("role")),
            
            columns=columns,
            rows=rows,
            
            markdown=markdown_val,
            html_content=html_val,
            
            source=source_val,
            method=_normalize_str(d.get("method")),
            numeric_trust=numeric_trust_val,
            
            structured_available=structured_avail,
            raw_available=raw_avail,
            structure_lossy=lossy,
            
            bbox=_safe_bbox(d.get("bbox")),
            extra=extra_data,
        )


# =============================================================================
# ImageBlock
# =============================================================================

@dataclass
class ImageBlock:
    """
    Represents an image extracted from the document.
    """
    id: str
    doc_id: str
    page: int
    file_path: str                 # Local or storage path
    caption: Optional[str] = None
    section: Optional[str] = None
    category: Optional[str] = None # e.g., "logo", "chart", "figure"
    role: Optional[str] = None     # e.g., "visual_data", "decorative"
    bbox: Optional[BBox] = None
    extra: Dict[str, Any] = field(default_factory=dict)

    # --- Backward Compatibility: image_path Alias ---
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
        d = _safe_dict(data)
        
        # Handle alias preference
        file_path = d.get("file_path") or d.get("image_path") or ""

        return cls(
            id=str(d.get("id", "")),
            doc_id=str(d.get("doc_id", "")),
            page=int(d.get("page", 1) or 1),
            file_path=str(file_path),
            caption=d.get("caption"),
            section=_normalize_str(d.get("section")),
            category=_normalize_str(d.get("category")),
            role=_normalize_str(d.get("role")),
            bbox=_safe_bbox(d.get("bbox")),
            extra=_safe_dict(d.get("extra")),
        )


# =============================================================================
# IngestedDocument
# =============================================================================

TIngested = TypeVar("TIngested", bound="IngestedDocument")

@dataclass
class IngestedDocument:
    """
    Root container for a fully processed document.
    """
    metadata: DocumentMetadata
    texts: List[TextBlock] = field(default_factory=list)
    tables: List[TableBlock] = field(default_factory=list)
    images: List[ImageBlock] = field(default_factory=list)
    schema_version: str = "1.0"

    def to_dict(self) -> Dict[str, Any]:
        """Full serialization."""
        return {
            "metadata": self.metadata.to_dict(),
            "texts": [t.to_dict() for t in self.texts],
            "tables": [tb.to_dict() for tb in self.tables],
            "images": [im.to_dict() for im in self.images],
            "schema_version": self.schema_version,
        }

    @classmethod
    def from_dict(cls: Type[TIngested], data: Dict[str, Any]) -> TIngested:
        d = _safe_dict(data)

        # Tolerant loading: Default to empty lists if keys missing
        meta_raw = d.get("metadata") or {}
        texts_raw = _safe_list(d.get("texts"))
        tables_raw = _safe_list(d.get("tables"))
        images_raw = _safe_list(d.get("images"))
        version = str(d.get("schema_version", "1.0"))

        return cls(
            metadata=DocumentMetadata.from_dict(meta_raw),
            texts=[TextBlock.from_dict(t) for t in texts_raw],
            tables=[TableBlock.from_dict(tb) for tb in tables_raw],
            images=[ImageBlock.from_dict(im) for im in images_raw],
            schema_version=version,
        )