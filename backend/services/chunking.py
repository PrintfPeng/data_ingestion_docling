# backend/services/chunking.py

from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional, Set, Tuple
import re
import hashlib 
from pydantic import BaseModel, Field
from ..models import (
    DocumentBundle,
    TableItem,
    TextBlock,
)

# PyThaiNLP Support
try:
    from pythainlp import sent_tokenize
    _HAS_PYTHAINLP = True
except ImportError:
    _HAS_PYTHAINLP = False

# --- Configuration (OPTIMIZED) ---
_TARGET_TOKENS = 300
_MAX_CHUNK_CHARS = 1200
_CHUNK_OVERLAP = 150
_MAX_INTENTS = 5

# --- [PATCH 1] Intent Priority Constant ---
_INTENT_PRIORITY = {
    "troubleshooting": 5,
    "safety": 5,
    "installation": 4,
    "financial": 3,
    "identity": 3,
    "reference": 2,
    "general": 1,
}

# --- Precompiled Regex Patterns (PERFORMANCE & SAFETY) ---
# Intent Keywords
_KW_INSTALL = r"(?:วิธี|ขั้นตอน|how\s*to|install|setup|การติดตั้ง|วิธีการ)"
_KW_TROUBLESHOOT = r"(?:แก้ปัญหา|error|fail|not\s*working|เสีย|ซ่อม|troubleshoot)"
_KW_SAFETY = r"(?:ความปลอดภัย|warning|danger|ระวัง|ห้าม|อันตราย)"
_KW_REF = r"(?:ความหมาย|คือ|definition|spec|สเปค|คุณลักษณะ)"
_KW_FINANCE = r"(?:ราคา|ค่าใช้จ่าย|เงิน|บาท|cost|price)"
_KW_IDENTITY = r"(?:ผู้|ชื่อ|ลงนาม|อนุมัติ|who|name|signature)"

# Scope Keywords
_KW_SCOPE_PROC = r"(?:step|ขั้นตอนที่|\d+\.)"
_KW_SCOPE_WARN = r"(?:warning|คำเตือน)"
_KW_SCOPE_TABLE = r"(?:table|ตาราง)"
_KW_SCOPE_EXAMPLE = r"(?:ตัวอย่าง|example|กรณี)"

# Entity Patterns
_RE_MONEY = re.compile(r'(\d+(?:,\d{3})*(?:\.\d{2})?)\s*(?:บาท|baht|฿)', re.IGNORECASE)
_RE_YEAR = re.compile(r'(?:ปี\s*)?(\d{4}|พ\.ศ\.\s*\d{4})', re.IGNORECASE)
_RE_THAI_NAME = re.compile(r'(?:นาย|นาง|นางสาว|คุณ|ดร\.|ศ\.|รศ\.|ผศ\.)\s*[\u0E00-\u0E7F]+\s+[\u0E00-\u0E7F]+', re.IGNORECASE)
_RE_HAS_NUM = re.compile(r'\d+')
_RE_QNA = re.compile(r'(?:ถาม|q|question)\s*[:\-]', re.IGNORECASE)

# Sanitization
_RE_SCRIPT = re.compile(r"<script.*?>.*?</script>", re.IGNORECASE | re.DOTALL)
_RE_JS_EVENT = re.compile(r" on\w+=", re.IGNORECASE)
_RE_JS_PROTO = re.compile(r"javascript:", re.IGNORECASE)
_RE_ZERO_WIDTH = re.compile(r"[\u200b\u200c\u200d\ufeff]")
_RE_MULTI_NEWLINE = re.compile(r"\n{3,}")
_RE_MULTI_SPACE = re.compile(r" {2,}")
_RE_MEANINGFUL = re.compile(r'[\w\u0E00-\u0E7F]{3,}')


class Chunk(BaseModel):
    id: str
    doc_id: str
    doc_type: str
    source: Literal["text", "table", "image"]
    page: Optional[int] = None
    content: str
    metadata: Dict[str, Any] = Field(default_factory=dict)


# -------------------------------------------------------------------
# Helper: Metadata Enrichment (IMPROVED - More Intelligent)
# -------------------------------------------------------------------
def _extract_intent_and_entities(text: str, section: str) -> Dict[str, Any]:
    """
    Analyzes text to create metadata for Filter/Boost.
    Robust against None inputs and regex failures.
    """
    # Defensive casting
    text_safe = str(text or "")
    section_safe = str(section or "")
    
    # Normalize for matching
    text_lower = text_safe.lower()
    section_lower = section_safe.lower()
    combined = f"{text_lower} {section_lower}"

    # 1. Detect Intent with Priority
    intent_scores: Dict[str, int] = {}
    
    if re.search(_KW_TROUBLESHOOT, combined):
        intent_scores["troubleshooting"] = 3
    if re.search(_KW_SAFETY, combined):
        intent_scores["safety"] = 3
    if re.search(_KW_INSTALL, combined):
        intent_scores["installation"] = 2
    if re.search(_KW_IDENTITY, combined):
        intent_scores["identity"] = 2
    if re.search(_KW_FINANCE, combined):
        intent_scores["financial"] = 2
    if re.search(_KW_REF, combined):
        intent_scores["reference"] = 1
    
    # Sort and Deduplicate
    intents = []
    if intent_scores:
        sorted_intents = sorted(intent_scores.items(), key=lambda x: x[1], reverse=True)
        intents = [intent for intent, _ in sorted_intents]
    else:
        intents = ["general"]

    # Cap intent list
    intents = intents[:_MAX_INTENTS]

    # 2. Detect Scope
    scope = "general"
    if re.search(_KW_SCOPE_PROC, combined):
        scope = "procedure"
    elif re.search(_KW_SCOPE_WARN, combined):
        scope = "warning"
    elif re.search(_KW_SCOPE_TABLE, combined):
        scope = "tabular"
    elif re.search(_KW_SCOPE_EXAMPLE, combined):
        scope = "example"

    # 3. Detect Entities (Safe Regex)
    entities = []
    
    # Limit entity extraction scope to avoid performance issues on huge strings
    search_text = combined[:5000] # Analyze first 5000 chars for metadata

    try:
        entities.extend([m.group(0) for m in _RE_MONEY.finditer(search_text)])
        entities.extend([m.group(0) for m in _RE_YEAR.finditer(search_text)])
        entities.extend([m.group(0) for m in _RE_THAI_NAME.finditer(search_text)])
    except Exception:
        # Fallback for regex safety
        pass

    unique_entities = sorted(list(set(entities)))[:10]  # Limit number of entities
    
    # [PATCH 1] Use Priority Selection for Primary Intent
    primary_intent = _select_primary_intent(intents)

    return {
        "intent": intents,
        "primary_intent": primary_intent,
        "answer_scope": scope,
        "entities": unique_entities,
        "has_numbers": bool(_RE_HAS_NUM.search(text_safe)),
        "has_names": bool(_RE_THAI_NAME.search(text_safe)),
    }

# [PATCH 1] Helper to deterministically select primary intent
def _select_primary_intent(intents: List[str]) -> str:
    if not intents:
        return "general"
    # Sort by Priority Descending, then Alphabetical (for stability)
    return sorted(
        intents,
        key=lambda x: (_INTENT_PRIORITY.get(x, 0), x),
        reverse=True
    )[0]

# -------------------------------------------------------------------
# Helper: Sanitization
# -------------------------------------------------------------------
def _sanitize_html_content(html_str: str) -> str:
    if not html_str:
        return ""
    try:
        clean = str(html_str)
        clean = _RE_SCRIPT.sub("", clean)
        clean = _RE_JS_EVENT.sub(" data-blocked-event=", clean)
        clean = _RE_JS_PROTO.sub("blocked:", clean)
        return clean.strip()
    except Exception:
        return ""


# -------------------------------------------------------------------
# Helper: Text Normalization (IMPROVED)
# -------------------------------------------------------------------
def _normalize_whitespace(text: str) -> str:
    if not text:
        return ""
    try:
        s = str(text)
        s = _RE_ZERO_WIDTH.sub("", s)
        s = s.replace("\xa0", " ")
        s = _RE_MULTI_NEWLINE.sub("\n\n", s)
        s = _RE_MULTI_SPACE.sub(" ", s)
        return s.strip()
    except Exception:
        return ""


def _has_meaningful_text(s: str) -> bool:
    if not s:
        return False
    s_safe = str(s).strip()
    return bool(_RE_MEANINGFUL.search(s_safe))


# -------------------------------------------------------------------
# [IMPROVED] Semantic Grouper - Smarter Context Detection
# -------------------------------------------------------------------
def _group_blocks_semantically(blocks: List[TextBlock]) -> List[Dict]:
    """
    Groups blocks semantically based on intent, section, and size.
    Prevents oversized chunks and ensures breaks on logical boundaries.
    """
    chunks = []
    current_chunk_blocks = []
    current_length = 0
    current_section = None
    current_intent_set: Set[str] = set()
    
    # [PATCH 2] Intent Cache to prevent drift/recalculation
    intent_cache: Dict[int, Dict] = {}

    for block in blocks:
        content = _normalize_whitespace(block.content)
        if not content or not _has_meaningful_text(content):
            continue

        # [PATCH 2] Cache Intent Extraction
        block_id = id(block)
        if block_id not in intent_cache:
            intent_cache[block_id] = _extract_intent_and_entities(content, block.section)
        
        block_meta = intent_cache[block_id]
        block_intent_set = set(block_meta["intent"])
        block_len = len(content)

        # Q&A Detection
        is_qna = bool(_RE_QNA.search(content))
        
        # Break Conditions
        is_new_section = (block.section != current_section) and current_chunk_blocks
        is_major_heading = block.extra.get("heading_level") == "H1"
        
        # [PATCH 3] Hard Stop Check (Pre-emptive)
        # If adding this block exceeds limit, we MUST break
        is_too_long = (current_length + block_len > _MAX_CHUNK_CHARS)
        
        # Intent Change Detection
        intent_changed = False
        if current_chunk_blocks:
            # 1. Disjoint intents suggest context switch
            if current_intent_set and block_intent_set:
                if current_intent_set.isdisjoint(block_intent_set):
                    intent_changed = True
            
            # 2. Priority intent drop (Troubleshoot -> General)
            # [PATCH 2] Use cached intent for last block
            last_block = current_chunk_blocks[-1]
            current_primary = intent_cache[id(last_block)]["primary_intent"]
            
            if current_primary in ["troubleshooting", "safety"]:
                if block_meta["primary_intent"] not in ["troubleshooting", "safety"]:
                    intent_changed = True

        should_break = is_new_section or is_too_long or is_major_heading or intent_changed or is_qna
        
        if should_break and current_chunk_blocks:
            # [PATCH 1] Deterministic Primary Intent
            chunks.append({
                "blocks": list(current_chunk_blocks),
                "section": current_section,
                "primary_intent": _select_primary_intent(list(current_intent_set))
            })
            
            # --- [FIX: CHUNK OVERLAP LOGIC] ---
            # ดึงข้อมูลจากส่วนท้ายของ Chunk ก่อนหน้ามาแปะใน Chunk ใหม่
            overlap_buffer = []
            overlap_len = 0
            
            # วิ่งย้อนหลังหา Block ที่ขนาดรวมกันไม่เกิน _CHUNK_OVERLAP
            for b in reversed(current_chunk_blocks):
                b_len = len(b.content)
                # ถ้าความยาวเกินลิมิต Overlap ที่ตั้งไว้ ให้หยุดดึง
                if overlap_len + b_len > _CHUNK_OVERLAP:
                    break
                overlap_buffer.insert(0, b) # ดันไปไว้ข้างหน้าเสมอเพื่อรักษาลำดับ
                overlap_len += b_len
            
            # นำ Overlap Buffer มาเป็นจุดเริ่มต้นของ Chunk ถัดไป
            current_chunk_blocks = list(overlap_buffer)
            current_length = overlap_len
            current_intent_set = set()
            
            # อัปเดต Intent ของบล็อกที่นำมาทำ Overlap อีกครั้ง
            for b in overlap_buffer:
                if id(b) in intent_cache:
                    current_intent_set.update(intent_cache[id(b)]["intent"])
            # ----------------------------------

        current_chunk_blocks.append(block)
        current_length += block_len
        current_section = block.section
        current_intent_set.update(block_intent_set)

    # Collect leftover
    if current_chunk_blocks:
        chunks.append({
            "blocks": list(current_chunk_blocks),
            "section": current_section,
            "primary_intent": _select_primary_intent(list(current_intent_set))
        })

    return chunks


def _format_chunk_content(group: Dict) -> Tuple[str, Dict]:
    """
    Assembles text content from blocks and generates metadata.
    Enforces length limits and safe serialization.
    """
    blocks: List[TextBlock] = group["blocks"]
    section = group.get("section") or "General"

    # Truncate Section Name if too long
    if len(section) > 50:
        section = section[:47] + "..."

    # Metadata Enrichment
    raw_text = "\n".join([b.content for b in blocks])
    # Ensure doc_id exists
    doc_id = blocks[0].doc_id if blocks else "unknown"
    semantic_meta = _extract_intent_and_entities(raw_text, section)

    # Content Assembly
    content_parts = []
    
    # Inject minimal safety tags based on intent
    if "safety" in semantic_meta["intent"]:
        content_parts.append("⚠️ [ข้อควรระวัง]")
    elif "troubleshooting" in semantic_meta["intent"]:
        content_parts.append("🔧 [การแก้ปัญหา]")
    
    page_numbers = set()
    block_types = set()

    for b in blocks:
        prefix = ""
        b_type = str(b.extra.get("block_type", "normal")).lower()
        
        if b_type == "warning":
            prefix = "⚠️ "
        elif b_type == "note":
            prefix = "ℹ️ "

        # [PATCH 5] Emoji Safety: Separate line to prevent LLM confusion
        if prefix:
            content_parts.append(prefix.strip())
        content_parts.append(b.content)

        if b.page:
            page_numbers.add(b.page)
        if b_type:
            block_types.add(b_type)

    full_content = "\n".join(content_parts)
    
    # Truncation (Safety Limit)
    if len(full_content) > _MAX_CHUNK_CHARS:
        full_content = full_content[:_MAX_CHUNK_CHARS - 50] + "\n...[ตัดทอนเนื้อหา]..."

    representative_page = min(page_numbers) if page_numbers else None
    
    dominant_type = "normal"
    if "warning" in block_types:
        dominant_type = "warning"
    elif "step" in block_types:
        dominant_type = "step"

    metadata = {
        "doc_id": str(doc_id),
        "page": representative_page,
        "pages": sorted(list(page_numbers))[:10], # Limit list size
        "section": section,
        "block_types": sorted(list(block_types))[:5],
        "dominant_block_type": dominant_type,
        "char_count": len(full_content),
        **semantic_meta,
        "source": "text"
    }

    return full_content, metadata


# -------------------------------------------------------------------
# 1) Text Chunking (OPTIMIZED)
# -------------------------------------------------------------------
def text_items_to_chunks(bundle: DocumentBundle) -> List[Chunk]:
    chunks: List[Chunk] = []

    # Filter valid texts
    valid_blocks = [t for t in bundle.texts if _has_meaningful_text(t.content)]
    if not valid_blocks:
        return chunks

    # Semantic Grouping
    grouped_chunks = _group_blocks_semantically(valid_blocks)

    # Deduplication
    seen_hashes = set()

    for group in grouped_chunks:
        content, meta = _format_chunk_content(group)
        if not content.strip():
            continue

        # [PATCH 4] Semantic Fingerprint (Content + Intent + Section)
        # Prevents collision if identical content appears in different contexts
        semantic_fingerprint = (
            content 
            + "|" + str(meta.get("primary_intent", "")) 
            + "|" + str(meta.get("section", ""))
        )
        content_hash = hashlib.md5(semantic_fingerprint.encode('utf-8', errors='ignore')).hexdigest()
        
        if content_hash in seen_hashes:
            continue
        seen_hashes.add(content_hash)

        # Stable Chunk ID
        chunk_id = f"{meta['doc_id']}::{content_hash[:8]}"
        doc_type = bundle.texts[0].doc_type if bundle.texts and bundle.texts[0].doc_type else "manual"

        chunks.append(
            Chunk(
                id=chunk_id,
                doc_id=str(meta["doc_id"]),
                doc_type=str(doc_type),
                source="text",
                page=meta["page"],
                content=content,
                metadata=meta,
            )
        )

    return chunks


# -------------------------------------------------------------------
# 2) Table Chunking (HYBRID INGESTION SUPPORT)
# -------------------------------------------------------------------

def _normalize_table_extra(item: TableItem) -> Dict[str, Any]:
    """
    Normalizes TableItem.extra with fallback priority:
    extra[key] -> item.<attr> -> default
    Ensures safe access to summary, category, markdown, html, role.
    Also handles 'image_path' and 'is_complex' for Hybrid Ingestion.
    """
    raw_extra = getattr(item, "extra", {}) or {}
    if not isinstance(raw_extra, dict):
        raw_extra = {}

    # 1. Summary
    summary = raw_extra.get("summary")
    if not summary and hasattr(item, "summary"):
         summary = getattr(item, "summary", "")
    
    # 2. Category
    category = raw_extra.get("category")
    if not category:
        category = getattr(item, "category", None)
    if not category:
        category = "general"
        
    # 3. Markdown (Metadata only)
    markdown = raw_extra.get("markdown_content") or raw_extra.get("markdown")
    if not markdown and hasattr(item, "markdown"):
        markdown = getattr(item, "markdown", "")
        
    # 4. HTML (Metadata only)
    html = raw_extra.get("html_content") or raw_extra.get("html")
    if not html and hasattr(item, "html"):
        html = getattr(item, "html", "")
    
    # 5. Role
    role = raw_extra.get("role") or getattr(item, "role", None) or ""

    # [ADDED] 6. Hybrid Ingestion Support
    # Check item attribute first, then extra
    image_path = getattr(item, "image_path", None)
    if not image_path:
        image_path = raw_extra.get("image_path")
        
    is_complex = getattr(item, "is_complex", None)
    if is_complex is None:
        is_complex = raw_extra.get("is_complex", False)

    return {
        "summary": str(summary or "").strip(),
        "category": str(category).strip().lower(),
        "markdown_content": str(markdown).strip(), 
        "html_content": str(html).strip(),
        "role": str(role).strip().lower(),
        "image_path": image_path, # Path to crop image (if complex)
        "is_complex": bool(is_complex) # Flag
    }

def _generate_table_semantic_rows(table: TableItem) -> str:
    """Smart Row Sampling"""
    if not table.rows or not table.columns:
        return ""
    
    semantic_rows = []
    headers = [str(c) for c in table.columns]
    MAX_ROWS = 15
    
    for i, row in enumerate(table.rows[:MAX_ROWS]):
        if not row:
            continue
            
        cells = [str(c or "").strip() for c in row]
        if not any(cells):
            continue

        row_parts = []
        for j, cell in enumerate(cells):
            if not cell or len(cell) > 100:
                continue
            col = headers[j] if j < len(headers) else f"Col{j+1}"
            row_parts.append(f"{col}={cell}")
        
        if row_parts:
            semantic_rows.append(" | ".join(row_parts[:5]))

    if len(table.rows) > MAX_ROWS:
        semantic_rows.append(f"... และอีก {len(table.rows) - MAX_ROWS} รายการ")

    return "\n".join(semantic_rows)


def table_items_to_chunks(bundle: DocumentBundle) -> List[Chunk]:
    """
    Create ONLY ONE UNIFIED CHUNK per table.
    Fully isolated and deterministic logic.
    Supports Hybrid Ingestion (Text for search, Image for display).
    """
    chunks: List[Chunk] = []
    
    for item in bundle.tables:
        # Step 1 - Normalize
        norm_extra = _normalize_table_extra(item)
        
        # Extract normalized fields
        summary = norm_extra["summary"]
        category = norm_extra["category"]
        role = norm_extra["role"]
        markdown_raw = norm_extra["markdown_content"]
        html_raw = norm_extra["html_content"]
        image_path = norm_extra["image_path"]
        is_complex = norm_extra["is_complex"]
        
        item_doc_type = item.doc_type or "manual"
        
        # Step 2 - Sanitize & Cap Metadata
        safe_html = _sanitize_html_content(html_raw)
        safe_markdown = markdown_raw[:2000] # Cap markdown length for metadata
        
        # Step 3 - Content Construction (Summary + Rows)
        content_parts = []
        
        # 3.1 Header / Name
        if item.name:
            content_parts.append(f"📊 {item.name}")
        
        # 3.2 Category
        if category and category != "general":
            content_parts.append(f"ประเภท: {category}")
            
        # 3.3 Summary (Priority 1)
        if summary:
            # Truncate summary if extremely long to save tokens for rows
            if len(summary) > 300:
                 summary = summary[:297] + "..."
            content_parts.append(summary)
            
        # 3.4 Columns (if small table)
        if item.columns and len(item.columns) <= 10:
             cols = [str(c) for c in item.columns if c]
             if cols:
                 content_parts.append(f"คอลัมน์: {', '.join(cols)}")
        
        # 3.5 Semantic Rows (Priority 2)
        # Use fallback raw data if no markdown, OR if it's a complex table we still want some text searchability
        semantic_rows = _generate_table_semantic_rows(item)
        
        # If markdown exists, append it (it's best for LLM)
        if markdown_raw:
             # Truncate if super long
             display_md = markdown_raw if len(markdown_raw) < 1000 else markdown_raw[:1000] + "..."
             content_parts.append(f"\n[Markdown Data]\n{display_md}")
        elif semantic_rows:
            content_parts.append(f"\n[Row Data]\n{semantic_rows}")
        elif not summary:
            content_parts.append("ตารางข้อมูล (ไม่มีรายละเอียด)")
            
        unified_content = "\n".join(content_parts)
        
        # Step 4 - Safety Truncate
        if len(unified_content) > _MAX_CHUNK_CHARS:
             unified_content = unified_content[:_MAX_CHUNK_CHARS - 50] + "\n...[ตัดทอนตาราง]..."

        # Intent Detection on the full content
        combined_for_intent = f"{item.name or ''}\n{summary}\n{unified_content}"
        semantic_meta = _extract_intent_and_entities(combined_for_intent, category)

        # Step 5 - Metadata Construction
        metadata = {
            "table_id": item.id,
            "doc_id": str(item.doc_id),
            "page": item.page,
            "columns": list(item.columns) if item.columns else [],
            "has_summary": bool(summary),
            "html_content": safe_html,       # Stored in metadata ONLY
            "markdown_content": safe_markdown, # Stored in metadata ONLY
            "category": category,
            "role": role,
            "html_trusted": False,
            "source": "table",
            "image_path": image_path,     # [CRITICAL] For Frontend Display
            "is_complex": is_complex,     # [CRITICAL] Flag for Hybrid Logic
            **semantic_meta,
        }

        # Step 6 - One Chunk per Table
        chunks.append(
            Chunk(
                id=f"{item.doc_id}::table::{item.id}",
                doc_id=str(item.doc_id),
                doc_type=str(item_doc_type),
                source="table",
                page=item.page,
                content=unified_content,
                metadata=metadata,
            )
        )

    return chunks


# -------------------------------------------------------------------
# 3) Image Chunking (IMPROVED)
# -------------------------------------------------------------------
def image_items_to_chunks(bundle: DocumentBundle) -> List[Chunk]:
    chunks: List[Chunk] = []
    
    for item in bundle.images:
        content = _normalize_whitespace(item.caption or "")
        if not content or not _has_meaningful_text(content):
            continue

        item_doc_type = item.doc_type or "manual"
        semantic_meta = _extract_intent_and_entities(content, "Image")

        clean_path = str(item.file_path or "").replace("\\", "/")
        formatted_content = (
            f"🖼️ [Image Info]\n"
            f"Path: {clean_path}\n"
            f"Page: {item.page or '?'}\n"
            f"Description: {content}"
        )

        chunks.append(
            Chunk(
                id=f"{item.doc_id}::image::{item.id}",
                doc_id=str(item.doc_id),
                doc_type=str(item_doc_type),
                source="image",
                page=item.page,
                content=formatted_content,
                metadata={
                    "image_id": item.id,
                    "file_path": str(item.file_path or ""),
                    "doc_id": str(item.doc_id),
                    "page": item.page,
                    "source": "image",
                    **semantic_meta,
                },
            )
        )
    
    return chunks