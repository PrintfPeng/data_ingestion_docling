from __future__ import annotations

"""
pdf_parser.py (Enterprise Manual RAG Edition)

หน้าที่:
- Extract Text + BBox อย่างแม่นยำ
- Layout Analysis: จัดการ Header/Footer และลำดับการอ่าน (2-Column)
- Structure Awareness: แยก H1/H2 และ Propagate Section
- Semantic Tagging: ระบุ Warning/Note/Step
- [NEW] Rich Metadata: เพิ่ม Intent Hint, Answer Scope, และ Entities ตั้งแต่ระดับ Block
- [NEW] Intelligent Merging: รวม Block โดยรักษา Metadata ให้ไม่หาย
- [NEW] Table Extraction & Embedding: ดึงตารางและฝังลง Vector DB ทันที
"""

from datetime import datetime
from pathlib import Path
from typing import List, Optional, Tuple, Set
import sys
import logging
import re
import statistics

import fitz  # PyMuPDF

from .schema import (
    DocumentMetadata,
    TextBlock,
    IngestedDocument,
    BBox,
)

# [NEW] Imports for Table Handling
from .table_extractor import extract_tables, table_to_text

# ==============================================================================
# [CRITICAL FIX] Correct Import for backend/services/vector_store.py
# ==============================================================================
try:
    # 1. ลอง Import ตาม Path ที่ถูกต้อง: backend.services.vector_store
    from backend.services.vector_store import get_vector_store
except ImportError:
    # 2. ถ้าไม่เจอ (อาจเพราะรัน script แล้ว python path ไม่ถึง root)
    # ให้ถอยกลับไปหา Root Folder (AI_Data_Ingestion) แล้ว Add เข้า sys.path
    try:
        current_file = Path(__file__).resolve()
        # ingestion/pdf_parser.py -> parent=ingestion -> parent=AI_Data_Ingestion (Root)
        project_root = current_file.parents[1]
        
        if str(project_root) not in sys.path:
            sys.path.append(str(project_root))
            
        from backend.services.vector_store import get_vector_store
    except ImportError as e:
        raise ImportError(
            f"CRITICAL ERROR: ไม่สามารถ Import 'backend.services.vector_store' ได้\n"
            f"ตำแหน่งไฟล์ที่คาดหวัง: {project_root / 'backend/services/vector_store.py'}\n"
            f"Error details: {e}"
        )

logger = logging.getLogger(__name__)


# -------------------------------------------------------------------
# 1. Config & Text Normalization
# -------------------------------------------------------------------
def _generate_doc_id(file_path: Path) -> str:
    # แทนที่เว้นวรรคด้วย _ เพื่อความชัวร์
    return file_path.stem.replace(" ", "_").replace("-", "_")

_WORD_CHARS_PATTERN = re.compile(r"[A-Za-z0-9\u0E00-\u0E7F]")

def _clean_text(text: str) -> str:
    if not text: return ""
    # ลบ control chars แต่เก็บ newline ไว้บางส่วนถ้าจำเป็น (ในที่นี้เรา merge แล้วจัดการทีหลัง)
    text = "".join(ch for ch in text if ch == "\n" or ch.isprintable())
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r" *\n *", "\n", text)
    return text.strip()

def _is_meaningful_text(text: str) -> bool:
    if not text: return False
    matches = _WORD_CHARS_PATTERN.findall(text)
    if len(matches) < 2: return False
    return True

def _normalize_section_title(text: str) -> str:
    text = text.strip()
    # Remove leading numbering like "1.", "1.1", "A)"
    text = re.sub(r"^(\d+(\.\d+)*|[A-Z])[\.\)]\s*", "", text)
    text = re.sub(r"\s+", " ", text)
    return text[:150]

# -------------------------------------------------------------------
# 2. Advanced Semantic Analysis (Metadata Injection)
# -------------------------------------------------------------------

# Keyword lists for Manual Domain (สามารถรับจาก config ภายนอกได้)
_INTENT_KEYWORDS = {
    "installation": ["install", "setup", "mounting", "connection", "wiring", "การติดตั้ง", "ต่อสาย"],
    "operation": ["operate", "use", "function", "start", "stop", "การใช้งาน", "วิธีใช้"],
    "troubleshooting": ["error", "fault", "problem", "solution", "fix", "troubleshoot", "แก้ปัญหา", "อาการเสีย"],
    "maintenance": ["maintain", "clean", "replace", "check", "inspection", "บำรุงรักษา", "ทำความสะอาด"],
    "safety": ["safety", "warning", "caution", "danger", "hazard", "ความปลอดภัย", "อันตราย"],
    "specification": ["spec", "dimension", "weight", "voltage", "technical data", "ข้อมูลจำเพาะ"]
}

_ENTITY_KEYWORDS = [
    "power button", "led", "lcd", "battery", "fuse", "sensor", "switch", 
    "terminal", "cable", "motor", "pump", "valve", "controller"
] # ตัวอย่าง entity

def _detect_block_type(text: str) -> str:
    text_upper = text.upper()
    if re.match(r"^(WARNING|CAUTION|DANGER|คำเตือน|ข้อควรระวัง)[:\s]", text_upper):
        return "warning"
    if re.match(r"^(NOTE|NOTICE|IMPORTANT|หมายเหตุ|สำคัญ|ข้อสังเกต)[:\s]", text_upper):
        return "note"
    # Step detection: Numbered list or "Step X"
    if re.match(r"^(\d+\.|Step\s+\d+|ขั้นตอนที่\s+\d+|[A-Z]\.)\s", text, re.IGNORECASE):
        return "step"
    return "normal"

def _analyze_intent(text: str, section: str) -> List[str]:
    """เดา Intent จากเนื้อหาและชื่อ Section"""
    combined = (text + " " + (section or "")).lower()
    intents = []
    for intent, keywords in _INTENT_KEYWORDS.items():
        if any(k in combined for k in keywords):
            intents.append(intent)
    return list(set(intents))

def _extract_entities(text: str) -> List[str]:
    """ดึง Entity สำคัญ (Simple matching)"""
    text_lower = text.lower()
    found = []
    for entity in _ENTITY_KEYWORDS:
        if entity in text_lower:
            found.append(entity)
    return found

def _determine_answer_scope(block_type: str) -> str:
    """กำหนดรูปแบบคำตอบที่เหมาะสม"""
    if block_type == "step": return "procedure"
    if block_type == "warning": return "warning"
    if block_type == "note": return "note"
    return "general"

# -------------------------------------------------------------------
# 3. Layout Analysis & Sorting
# -------------------------------------------------------------------
def _detect_header_footer(blocks: List[dict], page_height: float) -> List[dict]:
    # ปรับ Threshold ให้ปลอดภัย (Top/Bottom 7%)
    HEADER_THRESH = page_height * 0.07
    FOOTER_THRESH = page_height * 0.93
    
    for b in blocks:
        x0, y0, x1, y1 = b.get("bbox", (0,0,0,0))
        if "extra" not in b: b["extra"] = {}
        
        is_header = y1 < HEADER_THRESH
        is_footer = y0 > FOOTER_THRESH
        
        b["extra"]["is_header"] = is_header
        b["extra"]["is_footer"] = is_footer
        
        # Mark noise immediately to exclude later
        if is_header or is_footer:
            b["extra"]["noise"] = True
            
    return blocks

def _sort_blocks_reading_order(blocks: List[dict]) -> List[dict]:
    """
    Sort blocks using Row-Major Order with Tolerance.
    (ช่วยให้อ่าน 2 Column ได้ถูกต้อง โดยการ group Y ที่ใกล้เคียงกัน)
    """
    # Round Y0 to nearest 12px to treat them as same 'line'
    return sorted(blocks, key=lambda b: (int(b["bbox"][1] / 12), b["bbox"][0]))

# -------------------------------------------------------------------
# 4. Smart Merge Logic (The "Senior" Part)
# -------------------------------------------------------------------
def _merge_text_blocks(blocks: List[TextBlock]) -> List[TextBlock]:
    """
    Merge adjacent blocks semantically:
    - Merge paragraphs (same font, close proximity)
    - Group step sequences (Step 1 -> Step 2)
    - Preserve and merge Metadata (Intent, Entities)
    """
    if not blocks: return []
    
    merged: List[TextBlock] = []
    current = blocks[0]
    
    for next_block in blocks[1:]:
        # 1. Check compatibility
        curr_type = current.extra.get("block_type", "normal")
        next_type = next_block.extra.get("block_type", "normal")
        
        # Geometry
        vertical_dist = next_block.bbox[1] - current.bbox[3]
        
        # Logic: Step Sequence
        is_step_sequence = (curr_type in ["step", "step_sequence"] and next_type == "step")
        
        # Logic: Paragraph Continuation
        curr_font = current.extra.get("font_size", 0)
        next_font = next_block.extra.get("font_size", 0)
        font_diff = abs(curr_font - next_font)
        
        is_paragraph = (
            curr_type == "normal" and next_type == "normal" and
            font_diff < 1.5 and 
            vertical_dist < 15.0 and 
            vertical_dist > -5.0
        )
        
        same_section = (current.section == next_block.section)

        if same_section and (is_step_sequence or is_paragraph):
            # --- MERGE OPERATION ---
            delimiter = "\n" if is_step_sequence else " "
            current.content += delimiter + next_block.content
            
            # Merge BBox
            current.bbox = (
                min(current.bbox[0], next_block.bbox[0]),
                min(current.bbox[1], next_block.bbox[1]),
                max(current.bbox[2], next_block.bbox[2]),
                max(current.bbox[3], next_block.bbox[3]),
            )
            
            # Update Block Type
            if is_step_sequence:
                current.extra["block_type"] = "step_sequence"
                current.extra["answer_scope"] = "procedure" # Update scope
            
            # Merge Metadata (Intent & Entities)
            # Combine lists and remove duplicates
            curr_intents = set(current.extra.get("intent", []))
            next_intents = set(next_block.extra.get("intent", []))
            current.extra["intent"] = list(curr_intents.union(next_intents))
            
            curr_entities = set(current.extra.get("entities", []))
            next_entities = set(next_block.extra.get("entities", []))
            current.extra["entities"] = list(curr_entities.union(next_entities))
                
        else:
            merged.append(current)
            current = next_block
            
    merged.append(current)
    return merged

# -------------------------------------------------------------------
# 5. Page Extraction Logic
# -------------------------------------------------------------------
def _extract_text_blocks_from_page(
    pdf_page: fitz.Page,
    doc_id: str,
    page_number: int,
    start_index: int = 0,
    current_section: Optional[str] = None
) -> Tuple[List[TextBlock], Optional[str]]:
    
    try:
        # ใช้ raw dict เพื่อ control เองทั้งหมด
        page_dict = pdf_page.get_text("dict", flags=fitz.TEXT_PRESERVE_IMAGES)
    except Exception as e:
        logger.warning(f"Page {page_number} dict extraction failed: {e}")
        # Fallback
        txt = _clean_text(pdf_page.get_text("text") or "")
        if not _is_meaningful_text(txt):
            return [], current_section
        return [
            TextBlock(
                id=f"txt_{start_index:04d}",
                doc_id=doc_id,
                page=page_number,
                content=txt,
                section=current_section,
                category="fallback",
                bbox=(0.0,0.0,0.0,0.0),
                extra={"noise": False, "block_type": "normal", "intent": [], "entities": []}
            )
        ], current_section

    raw_blocks = page_dict.get("blocks", []) or []
    
    # 1. Detect Header/Footer
    raw_blocks = _detect_header_footer(raw_blocks, pdf_page.rect.height)
    
    # 2. Robust Sort
    raw_blocks = _sort_blocks_reading_order(raw_blocks)

    # 3. Calculate Page Statistics
    font_sizes = []
    for b in raw_blocks:
        if b.get("type") != 0: continue
        for line in b.get("lines", []):
            for span in line.get("spans", []):
                if span.get("size"): font_sizes.append(span["size"])
    
    page_median_font = statistics.median(font_sizes) if font_sizes else 10.0

    text_blocks: List[TextBlock] = []
    current_index = start_index
    active_section = current_section

    # --- Phase 1: Extraction & Tagging ---
    temp_blocks: List[TextBlock] = []
    
    for block in raw_blocks:
        if block.get("type") != 0: continue # Skip images

        # Assemble text
        lines = block.get("lines", [])
        spans_text = []
        block_fonts = []
        for line in lines:
            for span in line.get("spans", []):
                t = span.get("text", "").strip()
                if t:
                    spans_text.append(t)
                    block_fonts.append(span.get("size", 0))
        
        content = _clean_text(" ".join(spans_text))
        
        # Filter Noise
        if not _is_meaningful_text(content): continue
        if block.get("extra", {}).get("noise", False): continue # Skip Header/Footer

        # Heading Detection
        avg_font = sum(block_fonts)/len(block_fonts) if block_fonts else 0
        is_heading = False
        heading_level = None
        
        # Heuristic: Larger font + Short text + Not just numbers
        if avg_font > page_median_font * 1.2 and len(content) < 200:
            if not re.match(r"^[\d\.\,\s]+$", content):
                is_heading = True
                heading_level = "H1" if avg_font > page_median_font * 1.5 else "H2"

        # Semantic Analysis
        block_type = "heading" if is_heading else _detect_block_type(content)
        
        # Section Propagation
        if is_heading:
            normalized_header = _normalize_section_title(content)
            if normalized_header: active_section = normalized_header

        # Rich Metadata Extraction (Intent, Entities, Scope)
        intents = _analyze_intent(content, active_section)
        entities = _extract_entities(content)
        answer_scope = _determine_answer_scope(block_type)

        current_index += 1
        x0, y0, x1, y1 = block.get("bbox", (0,0,0,0))
        
        tb = TextBlock(
            id=f"txt_{current_index:04d}",
            doc_id=doc_id,
            page=page_number,
            content=content,
            section=active_section,
            category=None,
            bbox=(float(x0), float(y0), float(x1), float(y1)),
            extra={
                "font_size": avg_font,
                "is_heading": is_heading,
                "heading_level": heading_level,
                "block_type": block_type,
                # New Metadata
                "intent": intents,
                "answer_scope": answer_scope,
                "entities": entities
            }
        )
        temp_blocks.append(tb)

    # --- Phase 2: Merge Blocks ---
    merged_blocks = _merge_text_blocks(temp_blocks)
    
    return merged_blocks, active_section


# -------------------------------------------------------------------
# Main Parse Function
# -------------------------------------------------------------------
def parse_pdf(
    file_path: str | Path,
    doc_type: str = "generic",
    doc_id: Optional[str] = None,
    source: str = "uploaded",
) -> IngestedDocument:
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"PDF file not found: {path}")

    logger.info(f"[pdf_parser] Processing: {path.name}")
    pdf_doc = fitz.open(path)

    try:
        if doc_id is None:
            doc_id = _generate_doc_id(path)

        metadata = DocumentMetadata(
            doc_id=doc_id,
            file_name=path.name,
            doc_type=doc_type,
            page_count=pdf_doc.page_count,
            ingested_at=datetime.utcnow().isoformat(),
            source=source,
        )

        all_text_blocks: List[TextBlock] = []
        current_index = 0
        current_active_section = None

        for page_index in range(pdf_doc.page_count):
            page = pdf_doc[page_index]
            
            page_blocks, next_section = _extract_text_blocks_from_page(
                pdf_page=page,
                doc_id=doc_id,
                page_number=page_index + 1,
                start_index=current_index,
                current_section=current_active_section
            )
            
            all_text_blocks.extend(page_blocks)
            current_index += len(page_blocks)
            current_active_section = next_section

        logger.info(f"[pdf_parser] Finished text extraction for {doc_id}: {len(all_text_blocks)} blocks.")

        # -----------------------------------------------------------
        # [NEW] Table Extraction & Embedding (Embed ทันที!)
        # -----------------------------------------------------------
        logger.info(f"[pdf_parser] Extracting tables for {doc_id}...")
        extracted_tables = extract_tables(
            file_path=path,
            doc_id=doc_id,
            doc_type=doc_type
        )
        
        if extracted_tables:
            logger.info(f"[pdf_parser] Found {len(extracted_tables)} tables. Embedding into Vector Store...")
            try:
                # เรียกใช้ get_vector_store ผ่าน global variable ที่เราเตรียมไว้
                vs = get_vector_store()
                
                for table in extracted_tables:
                    # 1. แปลง Table Object เป็น Text String (พร้อม Summary & Columns)
                    text = table_to_text(table)
                    
                    # 2. เตรียม Metadata ที่ใช้สำหรับ Filter ใน rag.py
                    metadata_dict = {
                        "doc_id": table.doc_id,
                        "page": table.page,
                        "source": "table",       # สำคัญ: เพื่อให้ rag.py รู้ว่าเป็น table
                        "category": table.category, # ใช้สำหรับ Filter ประเภทตาราง
                        "table_id": table.id,
                        "doc_type": doc_type
                    }
                    
                    # 3. Add to Vector Store (ใช้ add_texts)
                    vs.add_texts(
                        texts=[text],
                        metadatas=[metadata_dict],
                        ids=[f"{table.doc_id}_{table.id}"]
                    )
            except Exception as e:
                logger.error(f"[pdf_parser] Failed to embed tables: {e}")

        return IngestedDocument(
            metadata=metadata,
            texts=all_text_blocks,
            tables=extracted_tables, # [MODIFIED] Return extracted tables populated
            images=[],
        )

    finally:
        pdf_doc.close()

if __name__ == "__main__":
    import json
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("pdf_path")
    args = parser.parse_args()

    doc = parse_pdf(args.pdf_path)
    # Print sample output with new metadata
    print(json.dumps([b.dict() for b in doc.texts[:5]], ensure_ascii=False, indent=2))