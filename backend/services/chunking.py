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
_TARGET_TOKENS = 300  # [FIX] ลดลงจาก 400 เพื่อให้ LLM อ่านง่ายขึ้น
_MAX_CHUNK_CHARS = 1200  # [FIX] ลดลงจาก 2000 (Chunk เล็กลง = ความแม่นยำสูงขึ้น)
_CHUNK_OVERLAP = 150  # [FIX] ลดลงตาม


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
    วิเคราะห์ Text เพื่อสร้าง Metadata สำหรับ Filter/Boost
    [FIX] ปรับให้ Smart ขึ้นด้วยการดู Context และ Priority
    """
    text_lower = text.lower()
    section_lower = (section or "").lower()
    combined = f"{text_lower} {section_lower}"

    # 1. Detect Intent with Priority (บางคำสำคัญกว่า)
    intents = []
    intent_scores = {}
    
    # [FIX] ใช้ Score System แทน Boolean
    if any(k in combined for k in ["วิธี", "ขั้นตอน", "how to", "install", "setup", "การติดตั้ง", "วิธีการ"]):
        intent_scores["installation"] = 2
    if any(k in combined for k in ["แก้ปัญหา", "error", "fail", "not working", "เสีย", "ซ่อม", "troubleshoot"]):
        intent_scores["troubleshooting"] = 3  # สูงสุดเพราะเป็นคำถามบ่อย
    if any(k in combined for k in ["ความปลอดภัย", "warning", "danger", "ระวัง", "ห้าม", "อันตราย"]):
        intent_scores["safety"] = 3
    if any(k in combined for k in ["ความหมาย", "คือ", "definition", "spec", "สเปค", "คุณลักษณะ"]):
        intent_scores["reference"] = 1
    if any(k in combined for k in ["ราคา", "ค่าใช้จ่าย", "เงิน", "บาท", "cost", "price"]):
        intent_scores["financial"] = 2
    if any(k in combined for k in ["ผู้", "ชื่อ", "ลงนาม", "อนุมัติ", "who", "name", "signature"]):
        intent_scores["identity"] = 2
    
    # เรียงตาม Score
    if intent_scores:
        sorted_intents = sorted(intent_scores.items(), key=lambda x: x[1], reverse=True)
        intents = [intent for intent, _ in sorted_intents]
    else:
        intents = ["general"]

    # 2. Detect Scope (MORE GRANULAR)
    scope = "general"
    if "step" in combined or "ขั้นตอนที่" in combined or re.search(r'\d+\.', combined):
        scope = "procedure"
    elif "warning" in combined or "คำเตือน" in combined:
        scope = "warning"
    elif "table" in combined or "ตาราง" in combined:
        scope = "tabular"
    elif any(k in combined for k in ["ตัวอย่าง", "example", "กรณี"]):
        scope = "example"

    # 3. Detect Entities (SMARTER - Use Regex Pattern)
    entities = []
    
    # Financial Entities
    money_pattern = r'(\d+(?:,\d{3})*(?:\.\d{2})?)\s*(?:บาท|baht|฿)'
    entities.extend([m.group(0) for m in re.finditer(money_pattern, combined)])
    
    # Year Entities
    year_pattern = r'(?:ปี\s*)?(\d{4}|พ\.ศ\.\s*\d{4})'
    entities.extend([m.group(0) for m in re.finditer(year_pattern, combined)])
    
    # Name Entities (Thai Names Pattern - คำนำหน้า + ชื่อ)
    name_pattern = r'(?:นาย|นาง|นางสาว|คุณ|ดร\.|ศ\.|รศ\.|ผศ\.)\s*[ก-๙]+\s+[ก-๙]+'
    entities.extend([m.group(0) for m in re.finditer(name_pattern, combined)])

    return {
        "intent": intents,
        "primary_intent": intents[0] if intents else "general",
        "answer_scope": scope,
        "entities": list(set(entities)),  # Remove duplicates
        "has_numbers": bool(re.search(r'\d+', text)),
        "has_names": bool(re.search(name_pattern, text, re.IGNORECASE)),
    }


# -------------------------------------------------------------------
# Helper: Sanitization
# -------------------------------------------------------------------
def _sanitize_html_content(html_str: str) -> str:
    if not html_str:
        return ""
    clean = re.sub(r"<script.*?>.*?</script>", "", html_str, flags=re.IGNORECASE | re.DOTALL)
    clean = re.sub(r" on\w+=", " data-blocked-event=", clean, flags=re.IGNORECASE)
    clean = re.sub(r"javascript:", "blocked:", clean, flags=re.IGNORECASE)
    return clean.strip()


# -------------------------------------------------------------------
# Helper: Text Normalization (IMPROVED)
# -------------------------------------------------------------------
def _normalize_whitespace(text: str) -> str:
    if not text:
        return ""
    # Remove zero-width characters
    text = re.sub(r"[\u200b\u200c\u200d\ufeff]", "", text)
    text = text.replace("\xa0", " ")
    # [FIX] Better newline handling
    text = re.sub(r"\n{3,}", "\n\n", text)  # Max 2 newlines
    text = re.sub(r" {2,}", " ", text)  # Max 1 space
    return text.strip()


def _has_meaningful_text(s: str) -> bool:
    if not s:
        return False
    s = str(s).strip()
    # [FIX] ต้องมีอย่างน้อย 3 ตัวอักษร
    return bool(re.search(r'[\w\u0E00-\u0E7F]{3,}', s))


# -------------------------------------------------------------------
# [IMPROVED] Semantic Grouper - Smarter Context Detection
# -------------------------------------------------------------------
def _group_blocks_semantically(blocks: List[TextBlock]) -> List[Dict]:
    """
    [FIX] รวม Block ที่เกี่ยวข้องกันอย่างฉลาดขึ้น
    - ดู Intent Change
    - ดู Section Change
    - ดู Content Type Change
    - Prevent over-sized chunks
    """
    chunks = []
    current_chunk_blocks = []
    current_length = 0
    current_section = None
    current_intent_set = set()

    for block in blocks:
        content = block.content.strip()
        if not content or not _has_meaningful_text(content):
            continue

        # Extract Intent
        block_meta = _extract_intent_and_entities(content, block.section)
        block_intent_set = set(block_meta["intent"])
        block_len = len(content)

        # [FIX] เพิ่มเงื่อนไข: ถ้าเป็น Q&A Pattern ให้แยก Chunk (เพราะมักจะเป็น Standalone)
        is_qna = bool(re.search(r'(?:ถาม|q|question)\s*[:\-]', content, re.IGNORECASE))
        
        # Break Conditions (OPTIMIZED)
        is_new_section = (block.section != current_section) and current_chunk_blocks
        is_too_long = current_length + block_len > _MAX_CHUNK_CHARS
        is_major_heading = block.extra.get("heading_level") == "H1"
        
        # [FIX] Intent Change Detection - More Nuanced
        intent_changed = False
        if current_intent_set and block_intent_set:
            # ถ้า Intent ไม่มี Overlap เลย = บริบทเปลี่ยนแน่
            if current_intent_set.isdisjoint(block_intent_set):
                intent_changed = True
            # [NEW] หรือถ้า Primary Intent เปลี่ยนจาก High Priority -> Low Priority
            elif block_meta["primary_intent"] in ["troubleshooting", "safety"] and \
                 current_chunk_blocks and \
                 _extract_intent_and_entities(current_chunk_blocks[-1].content, current_chunk_blocks[-1].section)["primary_intent"] not in ["troubleshooting", "safety"]:
                intent_changed = True

        should_break = is_new_section or is_too_long or is_major_heading or intent_changed or is_qna
        
        if should_break and current_chunk_blocks:
            chunks.append({
                "blocks": current_chunk_blocks,
                "section": current_section,
                "primary_intent": list(current_intent_set)[0] if current_intent_set else "general"
            })
            current_chunk_blocks = []
            current_length = 0
            current_intent_set = set()

        current_chunk_blocks.append(block)
        current_length += block_len
        current_section = block.section
        current_intent_set.update(block_intent_set)

    # Collect leftover
    if current_chunk_blocks:
        chunks.append({
            "blocks": current_chunk_blocks,
            "section": current_section,
            "primary_intent": list(current_intent_set)[0] if current_intent_set else "general"
        })

    return chunks


def _format_chunk_content(group: Dict) -> Tuple[str, Dict]:
    """
    [FIX] แปลงกลุ่ม Block ให้เป็น Text - ลด Noise ลง
    """
    blocks: List[TextBlock] = group["blocks"]
    section = group["section"] or "General"

    # [FIX] Shorten Section Name ถ้ายาวเกิน 50 chars
    if len(section) > 50:
        section = section[:47] + "..."

    # Metadata Enrichment
    raw_text = "\n".join([b.content for b in blocks])
    doc_id = blocks[0].doc_id
    semantic_meta = _extract_intent_and_entities(raw_text, section)

    # [FIX] Content Assembly - MINIMAL INJECTION
    content_parts = []
    
    # [FIX] เอา Section ออกจาก Content (ใส่ใน Metadata อย่างเดียวพอ)
    # แต่ถ้า Intent เป็นพวก Safety/Troubleshooting ให้ใส่ Tag สั้นๆ เพื่อ Boost
    if "safety" in semantic_meta["intent"]:
        content_parts.append("⚠️ [ข้อควรระวัง]")
    elif "troubleshooting" in semantic_meta["intent"]:
        content_parts.append("🔧 [การแก้ปัญหา]")
    
    page_numbers = set()
    block_types = set()

    for b in blocks:
        # [FIX] ลด Prefix - ใช้แค่ Emoji
        prefix = ""
        b_type = b.extra.get("block_type", "normal")
        if b_type == "warning":
            prefix = "⚠️ "
        elif b_type == "note":
            prefix = "ℹ️ "

        content_parts.append(f"{prefix}{b.content}")

        if b.page:
            page_numbers.add(b.page)
        if b_type:
            block_types.add(b_type)

    full_content = "\n".join(content_parts)
    
    # [FIX] Truncation with better message
    if len(full_content) > _MAX_CHUNK_CHARS:
        full_content = full_content[:_MAX_CHUNK_CHARS - 50] + "\n...[ตัดทอนเนื้อหา]..."

    representative_page = min(page_numbers) if page_numbers else None
    dominant_type = "warning" if "warning" in block_types else ("step" if "step" in block_types else "normal")

    metadata = {
        "doc_id": str(doc_id),
        "page": representative_page,
        "pages": list(page_numbers),
        "section": section,
        "block_types": list(block_types),
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

    # [FIX] Deduplication - ใช้ Set เก็บ Content Hash
    seen_hashes = set()

    for i, group in enumerate(grouped_chunks):
        content, meta = _format_chunk_content(group)

        # [FIX] Check Duplication
        content_hash = hashlib.md5(content.encode('utf-8')).hexdigest()
        if content_hash in seen_hashes:
            continue  # Skip duplicate
        seen_hashes.add(content_hash)

        # Stable Chunk ID
        chunk_id = f"{meta['doc_id']}::{content_hash[:8]}"
        doc_type = bundle.texts[0].doc_type if bundle.texts and bundle.texts[0].doc_type else "manual"

        chunks.append(
            Chunk(
                id=chunk_id,
                doc_id=meta["doc_id"],
                doc_type=doc_type,
                source="text",
                page=meta["page"],
                content=content,
                metadata=meta,
            )
        )

    return chunks


# -------------------------------------------------------------------
# 2) Table Chunking (SIMPLIFIED - Less Redundancy)
# -------------------------------------------------------------------
def _generate_table_summary_text(table: TableItem, extra: Dict) -> str:
    """Compact Summary"""
    parts = []
    if table.name:
        parts.append(f"📊 {table.name}")
    
    category = extra.get("category") or getattr(table, "category", None)
    if category:
        parts.append(f"ประเภท: {category}")
    
    summary = extra.get("summary", "").strip()
    if summary:
        # [FIX] Truncate long summary
        if len(summary) > 200:
            summary = summary[:197] + "..."
        parts.append(summary)
    
    if table.columns and len(table.columns) <= 10:  # [FIX] เฉพาะตารางเล็ก
        parts.append(f"คอลัมน์: {', '.join(table.columns)}")
    
    return "\n".join(parts)


def _generate_table_semantic_rows(table: TableItem) -> str:
    """[FIX] Smart Row Sampling - ไม่ใส่ทุก Row"""
    if not table.rows or not table.columns:
        return ""
    
    semantic_rows = []
    headers = table.columns
    MAX_ROWS = 15  # [FIX] จำกัดจำนวน Row
    
    for i, row in enumerate(table.rows[:MAX_ROWS]):
        cells = [str(c or "").strip() for c in row]
        if not any(cells):
            continue

        # [FIX] แสดงแบบกระชับ
        row_parts = []
        for j, cell in enumerate(cells):
            if not cell or len(cell) > 100:  # [FIX] Skip long cells
                continue
            col = headers[j] if j < len(headers) else f"Col{j+1}"
            row_parts.append(f"{col}={cell}")
        
        if row_parts:
            semantic_rows.append(" | ".join(row_parts[:5]))  # [FIX] Max 5 columns per row

    if len(table.rows) > MAX_ROWS:
        semantic_rows.append(f"... และอีก {len(table.rows) - MAX_ROWS} รายการ")

    return "\n".join(semantic_rows)


def table_items_to_chunks(bundle: DocumentBundle) -> List[Chunk]:
    """[FIX] Create ONLY ONE UNIFIED CHUNK per table"""
    chunks: List[Chunk] = []
    
    for item in bundle.tables:
        extra = getattr(item, "extra", {}) or {}
        raw_html = extra.get("html_content", "")
        safe_html = _sanitize_html_content(raw_html)
        markdown_code = extra.get("markdown_content", "")
        category = extra.get("category") or getattr(item, "category", "")
        role = extra.get("role", "")
        item_doc_type = item.doc_type or "manual"

        # [FIX] Unified Content - รวมทุกอย่างใน 1 Chunk
        summary_text = _generate_table_summary_text(item, extra)
        semantic_text = _generate_table_semantic_rows(item)
        
        unified_content_parts = []
        if summary_text:
            unified_content_parts.append(summary_text)
        if semantic_text:
            unified_content_parts.append(f"\nข้อมูลตาราง:\n{semantic_text}")
        
        unified_content = "\n".join(unified_content_parts)
        
        if not unified_content.strip():
            continue  # Skip empty tables

        # Intent Detection
        combined_for_intent = f"{item.name or ''}\n{extra.get('summary','')}\n{unified_content}"
        semantic_meta = _extract_intent_and_entities(combined_for_intent, category)

        metadata = {
            "table_id": item.id,
            "doc_id": str(item.doc_id),
            "page": item.page,
            "columns": str(item.columns),
            "has_summary": bool(extra.get("summary")),
            "html_content": safe_html,
            "markdown_content": markdown_code,
            "category": category,
            "role": role,
            "html_trusted": False,
            "source": "table",
            **semantic_meta,
        }

        # [FIX] Single Unified Chunk
        chunks.append(
            Chunk(
                id=f"{item.doc_id}::table::{item.id}",
                doc_id=str(item.doc_id),
                doc_type=item_doc_type,
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

        # [FIX] เพิ่ม Context แบบกระชับ
        formatted_content = f"🖼️ [{item.page or '?'}] {content}"

        chunks.append(
            Chunk(
                id=f"{item.doc_id}::image::{item.id}",
                doc_id=str(item.doc_id),
                doc_type=item_doc_type,
                source="image",
                page=item.page,
                content=formatted_content,
                metadata={
                    "image_id": item.id,
                    "file_path": item.file_path,
                    "doc_id": str(item.doc_id),  # [CRITICAL FIX] ใส่ doc_id ใน metadata
                    "page": item.page,
                    "source": "image",
                    **semantic_meta,
                },
            )
        )
    
    return chunks