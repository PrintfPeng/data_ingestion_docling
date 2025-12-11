from __future__ import annotations

"""
validator.py

ชุดฟังก์ชันตรวจความถูกต้องของ IngestedDocument:

- validate_document_structure: ตรวจ metadata + โครงสร้างรวม ๆ
- validate_text_blocks: ตรวจ TextBlock
- validate_tables: ตรวจตาราง
- validate_images: ตรวจรูป
- validate_all: รวมทุกอย่างแล้วคืน issues เป็น list[dict]

ใช้สำหรับ:
- เช็คคุณภาพ ingestion
- log ปัญหาไว้ใน validation.json
"""

from typing import List, Dict, Any, Tuple, Optional

from .schema import IngestedDocument, TableBlock, ImageBlock, TextBlock


# -------------------------------------------------------------------
# Helper: issue factory
# -------------------------------------------------------------------


def _issue(
    level: str,
    code: str,
    message: str,
    context: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    return {
        "level": level,   # "info" | "warning" | "error"
        "code": code,
        "message": message,
        "context": context or {},
    }


# -------------------------------------------------------------------
# Helper: page / id statistics
# -------------------------------------------------------------------


def _collect_page_stats(doc: IngestedDocument) -> Tuple[Optional[int], Optional[int]]:
    """หาหน้า min/max ที่ปรากฏใน texts / tables / images"""
    pages: List[int] = []

    for t in doc.texts:
        p = getattr(t, "page", None)
        if isinstance(p, int):
            pages.append(p)

    for tb in doc.tables:
        p = getattr(tb, "page", None)
        if isinstance(p, int):
            pages.append(p)

    for im in doc.images:
        p = getattr(im, "page", None)
        if isinstance(p, int):
            pages.append(p)

    if not pages:
        return None, None

    return min(pages), max(pages)


def _collect_ids(items) -> List[str]:
    ids: List[str] = []
    for x in items:
        _id = getattr(x, "id", None)
        if isinstance(_id, str):
            ids.append(_id)
    return ids


# -------------------------------------------------------------------
# 1) Document-level validation
# -------------------------------------------------------------------


def validate_document_structure(doc: IngestedDocument) -> List[Dict[str, Any]]:
    issues: List[Dict[str, Any]] = []

    # metadata basic
    if not doc.metadata.doc_id:
        issues.append(
            _issue(
                "error",
                "MISSING_DOC_ID",
                "Document metadata.doc_id is empty.",
            )
        )

    if not doc.metadata.file_name:
        issues.append(
            _issue(
                "warning",
                "MISSING_FILE_NAME",
                "Document metadata.file_name is empty.",
            )
        )

    # page_count vs content pages
    min_page, max_page = _collect_page_stats(doc)
    meta_page_count = getattr(doc.metadata, "page_count", None)

    if meta_page_count is None:
        if max_page is not None:
            issues.append(
                _issue(
                    "warning",
                    "MISSING_PAGE_COUNT",
                    "Document metadata.page_count is missing but blocks have page info.",
                    {"min_page": min_page, "max_page": max_page},
                )
            )
    else:
        if meta_page_count <= 0:
            issues.append(
                _issue(
                    "warning",
                    "INVALID_PAGE_COUNT",
                    f"Document metadata.page_count={meta_page_count} is not positive.",
                    {"page_count": meta_page_count},
                )
            )
        if max_page is not None and max_page > meta_page_count:
            issues.append(
                _issue(
                    "warning",
                    "PAGE_COUNT_MISMATCH",
                    "Some blocks have page index greater than metadata.page_count.",
                    {"page_count": meta_page_count, "max_block_page": max_page},
                )
            )

    # presence of texts
    if not doc.texts:
        issues.append(
            _issue(
                "error",
                "NO_TEXT_BLOCKS",
                "Document has no TextBlock entries.",
            )
        )

    # duplicated ids
    text_ids = _collect_ids(doc.texts)
    table_ids = _collect_ids(doc.tables)
    image_ids = _collect_ids(doc.images)

    def _find_duplicates(ids: List[str]) -> List[str]:
        seen = set()
        dup = set()
        for i in ids:
            if i in seen:
                dup.add(i)
            else:
                seen.add(i)
        return list(dup)

    dup_text = _find_duplicates(text_ids)
    dup_table = _find_duplicates(table_ids)
    dup_image = _find_duplicates(image_ids)

    if dup_text:
        issues.append(
            _issue(
                "warning",
                "DUPLICATE_TEXT_ID",
                "Found duplicated TextBlock.id values.",
                {"ids": dup_text},
            )
        )
    if dup_table:
        issues.append(
            _issue(
                "warning",
                "DUPLICATE_TABLE_ID",
                "Found duplicated TableBlock.id values.",
                {"ids": dup_table},
            )
        )
    if dup_image:
        issues.append(
            _issue(
                "warning",
                "DUPLICATE_IMAGE_ID",
                "Found duplicated ImageBlock.id values.",
                {"ids": dup_image},
            )
        )

    return issues


# -------------------------------------------------------------------
# 2) TextBlock validation
# -------------------------------------------------------------------


def _validate_single_text_block(
    doc: IngestedDocument,
    block: TextBlock,
    index: int,
) -> List[Dict[str, Any]]:
    issues: List[Dict[str, Any]] = []

    meta_doc_id = doc.metadata.doc_id

    if block.doc_id and meta_doc_id and block.doc_id != meta_doc_id:
        issues.append(
            _issue(
                "warning",
                "TEXT_DOC_ID_MISMATCH",
                f"TextBlock index={index} doc_id='{block.doc_id}' != metadata.doc_id='{meta_doc_id}'.",
                {"index": index, "block_doc_id": block.doc_id, "meta_doc_id": meta_doc_id},
            )
        )

    # page range check
    page_count = getattr(doc.metadata, "page_count", None)
    page = getattr(block, "page", None)
    if isinstance(page, int):
        if page <= 0:
            issues.append(
                _issue(
                    "warning",
                    "TEXT_PAGE_INVALID",
                    f"TextBlock index={index} has non-positive page={page}.",
                    {"index": index, "page": page},
                )
            )
        if page_count is not None and page > page_count:
            issues.append(
                _issue(
                    "warning",
                    "TEXT_PAGE_OUT_OF_RANGE",
                    f"TextBlock index={index} has page={page} > page_count={page_count}.",
                    {"index": index, "page": page, "page_count": page_count},
                )
            )

    # content sanity
    content = block.content or ""
    if len(content) > 8000:
        issues.append(
            _issue(
                "info",
                "TEXT_BLOCK_VERY_LONG",
                f"TextBlock index={index} has very long content (len={len(content)}).",
                {"index": index, "length": len(content)},
            )
        )

    if len(content.strip()) < 2:
        issues.append(
            _issue(
                "info",
                "TEXT_BLOCK_VERY_SHORT",
                f"TextBlock index={index} has very short content.",
                {"index": index, "content": content},
            )
        )

    # bbox check (structural)
    bbox = getattr(block, "bbox", None)
    if bbox is not None:
        if not (isinstance(bbox, (list, tuple)) and len(bbox) == 4):
            issues.append(
                _issue(
                    "warning",
                    "TEXT_BBOX_INVALID",
                    f"TextBlock index={index} bbox is not a 4-tuple.",
                    {"index": index, "bbox": bbox},
                )
            )

    # section / role optional but useful
    extra = block.extra or {}
    if "section" not in extra:
        issues.append(
            _issue(
                "info",
                "TEXT_NO_SECTION",
                f"TextBlock index={index} has no section tag in extra['section'].",
                {"index": index},
            )
        )
    if "role" not in extra:
        issues.append(
            _issue(
                "info",
                "TEXT_NO_ROLE",
                f"TextBlock index={index} has no role tag in extra['role'].",
                {"index": index},
            )
        )

    return issues


def validate_text_blocks(doc: IngestedDocument) -> List[Dict[str, Any]]:
    issues: List[Dict[str, Any]] = []

    for idx, block in enumerate(doc.texts):
        issues.extend(_validate_single_text_block(doc, block, idx))

    return issues


# -------------------------------------------------------------------
# 3) TableBlock validation
# -------------------------------------------------------------------


def validate_tables(doc: IngestedDocument) -> List[Dict[str, Any]]:
    issues: List[Dict[str, Any]] = []

    meta_doc_id = doc.metadata.doc_id
    page_count = getattr(doc.metadata, "page_count", None)

    for idx, tb in enumerate(doc.tables):
        header = getattr(tb, "header", [])
        rows = getattr(tb, "rows", [])

        # doc_id consistency
        if tb.doc_id and meta_doc_id and tb.doc_id != meta_doc_id:
            issues.append(
                _issue(
                    "warning",
                    "TABLE_DOC_ID_MISMATCH",
                    f"Table index={idx} doc_id='{tb.doc_id}' != metadata.doc_id='{meta_doc_id}'.",
                    {"table_index": idx, "table_doc_id": tb.doc_id, "meta_doc_id": meta_doc_id},
                )
            )

        # page range
        page = getattr(tb, "page", None)
        if isinstance(page, int):
            if page <= 0:
                issues.append(
                    _issue(
                        "warning",
                        "TABLE_PAGE_INVALID",
                        f"Table index={idx} has non-positive page={page}.",
                        {"table_index": idx, "page": page},
                    )
                )
            if page_count is not None and page > page_count:
                issues.append(
                    _issue(
                        "warning",
                        "TABLE_PAGE_OUT_OF_RANGE",
                        f"Table index={idx} has page={page} > page_count={page_count}.",
                        {"table_index": idx, "page": page, "page_count": page_count},
                    )
                )

        # header / rows presence
        if not header and rows:
            issues.append(
                _issue(
                    "warning",
                    "TABLE_NO_HEADER",
                    f"Table index={idx} has rows but empty header.",
                    {"table_index": idx},
                )
            )

        if header and not rows:
            issues.append(
                _issue(
                    "warning",
                    "TABLE_NO_ROWS",
                    f"Table index={idx} has header but no rows.",
                    {"table_index": idx},
                )
            )

        # header/rows length mismatch
        for r_idx, row in enumerate(rows):
            if header and len(row) != len(header):
                issues.append(
                    _issue(
                        "warning",
                        "ROW_LEN_MISMATCH",
                        (
                            f"Table index={idx} row={r_idx} "
                            f"len(row)={len(row)} != len(header)={len(header)}"
                        ),
                        {"table_index": idx, "row_index": r_idx},
                    )
                )

        # bbox check
        bbox = getattr(tb, "bbox", None)
        if bbox is not None:
            if not (isinstance(bbox, (list, tuple)) and len(bbox) == 4):
                issues.append(
                    _issue(
                        "warning",
                        "TABLE_BBOX_INVALID",
                        f"Table index={idx} bbox is not a 4-tuple.",
                        {"table_index": idx, "bbox": bbox},
                    )
                )

        # category / role hints
        extra = tb.extra or {}
        if not getattr(tb, "category", None):
            issues.append(
                _issue(
                    "info",
                    "TABLE_NO_CATEGORY",
                    f"Table index={idx} has no category.",
                    {"table_index": idx},
                )
            )
        if "role" not in extra:
            issues.append(
                _issue(
                    "info",
                    "TABLE_NO_ROLE",
                    f"Table index={idx} has no role in extra['role'].",
                    {"table_index": idx},
                )
            )

    return issues


# -------------------------------------------------------------------
# 4) ImageBlock validation
# -------------------------------------------------------------------


def validate_images(doc: IngestedDocument) -> List[Dict[str, Any]]:
    """
    ตรวจ ImageBlock แบบเบา ๆ:
    - ต้องมีอย่างน้อยหนึ่งใน image_path / file_path / ref
    - page อยู่ในช่วง
    """
    issues: List[Dict[str, Any]] = []

    meta_doc_id = doc.metadata.doc_id
    page_count = getattr(doc.metadata, "page_count", None)

    for idx, im in enumerate(doc.images):
        # doc_id consistency ถ้ามี field
        im_doc_id = getattr(im, "doc_id", None)
        if im_doc_id and meta_doc_id and im_doc_id != meta_doc_id:
            issues.append(
                _issue(
                    "warning",
                    "IMAGE_DOC_ID_MISMATCH",
                    f"Image index={idx} doc_id='{im_doc_id}' != metadata.doc_id='{meta_doc_id}'.",
                    {"image_index": idx, "image_doc_id": im_doc_id, "meta_doc_id": meta_doc_id},
                )
            )

        path = getattr(im, "image_path", None) or getattr(im, "file_path", None)
        ref = getattr(im, "ref", None)

        if not path and not ref:
            issues.append(
                _issue(
                    "warning",
                    "IMAGE_NO_PATH",
                    f"Image index={idx} has no image_path/file_path/ref.",
                    {"image_index": idx},
                )
            )

        # page range
        page = getattr(im, "page", None)
        if isinstance(page, int):
            if page <= 0:
                issues.append(
                    _issue(
                        "warning",
                        "IMAGE_PAGE_INVALID",
                        f"Image index={idx} has non-positive page={page}.",
                        {"image_index": idx, "page": page},
                    )
                )
            if page_count is not None and page > page_count:
                issues.append(
                    _issue(
                        "warning",
                        "IMAGE_PAGE_OUT_OF_RANGE",
                        f"Image index={idx} has page={page} > page_count={page_count}.",
                        {"image_index": idx, "page": page, "page_count": page_count},
                    )
                )

    return issues


# -------------------------------------------------------------------
# 5) validate_all
# -------------------------------------------------------------------


def validate_all(doc: IngestedDocument) -> List[Dict[str, Any]]:
    """
    รวบทุก validation:
    - document structure
    - text blocks
    - tables
    - images
    """
    issues: List[Dict[str, Any]] = []
    issues.extend(validate_document_structure(doc))
    issues.extend(validate_text_blocks(doc))
    issues.extend(validate_tables(doc))
    issues.extend(validate_images(doc))
    return issues
