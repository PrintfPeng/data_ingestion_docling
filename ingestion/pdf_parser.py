from __future__ import annotations

"""
pdf_parser.py (Final Master Edition)

หน้าที่:
- Extract Text + BBox อย่างแม่นยำ (รักษา Logic การรวมบรรทัดและการวิเคราะห์ Intent)
- Layout Analysis: จัดการ Header/Footer
- Semantic Tagging: ระบุ Warning/Note/Step
- Table Extraction: ดึงตารางและฝังลง Vector DB ทันที
- Image Extraction: ดึงรูปภาพประกอบ (Integrated)
- Markdown Generation: สร้างรายงานสรุปไฟล์ .md พร้อมรูปภาพและตาราง (Integrated)
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

# ------------------------------------------------------------------
# IMPORTS: External Extractors & Generators
# ------------------------------------------------------------------
# ต้องมั่นใจว่าไฟล์เหล่านี้มีอยู่จริงใน folder ingestion/
try:
    from .table_extractor import extract_tables, table_to_text
    from .image_extractor import extract_images       # ต้องมีไฟล์ image_extractor.py
    from .markdown_generator import generate_markdown # ต้องมีไฟล์ markdown_generator.py
except ImportError as e:
    # แจ้งเตือนถ้าไฟล์ย่อยหายไป แต่ไม่ให้โปรแกรม Crash ทันที (จะ error ตอนรันฟังก์ชันนั้นๆ แทน)
    logging.warning(f"[pdf_parser] Dependency import warning: {e}")
    # Dummy functions เพื่อกัน Crash ตอน import
    if 'extract_tables' not in locals(): extract_tables = lambda *a, **k: []
    if 'table_to_text' not in locals(): table_to_text = lambda x: ""
    if 'extract_images' not in locals(): extract_images = lambda *a, **k: []
    if 'generate_markdown' not in locals(): generate_markdown = lambda *a, **k: None

# ------------------------------------------------------------------
# CRITICAL FIX: Import Vector Store
# ------------------------------------------------------------------
try:
    from backend.services.vector_store import get_vector_store
except ImportError:
    try:
        current_file = Path(__file__).resolve()
        project_root = current_file.parents[1]
        if str(project_root) not in sys.path:
            sys.path.append(str(project_root))
        from backend.services.vector_store import get_vector_store
    except ImportError as e:
        raise ImportError(
            f"CRITICAL ERROR: ไม่สามารถ Import 'backend.services.vector_store' ได้\n"
            f"Error details: {e}"
        )

logger = logging.getLogger(__name__)


# ==============================================================================
# 1. Config & Text Normalization Helpers (Logic เดิม - ห้ามตัด)
# ==============================================================================
def _generate_doc_id(file_path: Path) -> str:
    return file_path.stem.replace(" ", "_").replace("-", "_")

_WORD_CHARS_PATTERN = re.compile(r"[A-Za-z0-9\u0E00-\u0E7F]")

def _clean_text(text: str) -> str:
    if not text: return ""
    # ลบ control chars แต่เก็บ newline
    text = "".join(ch for ch in text if ch == "\n" or ch.isprintable())
    # ยุบ space เกิน
    text = re.sub(r"[ \t]+", " ", text)
    # ยุบ newline เกิน
    text = re.sub(r" *\n *", "\n", text)
    return text.strip()

def _is_meaningful_text(text: str) -> bool:
    if not text: return False
    matches = _WORD_CHARS_PATTERN.findall(text)
    if len(matches) < 2: return False
    return True

def _normalize_section_title(text: str) -> str:
    text = text.strip()
    # ลบเลขนำหน้า เช่น "1. Introduction" -> "Introduction"
    text = re.sub(r"^(\d+(\.\d+)*|[A-Z])[\.\)]\s*", "", text)
    text = re.sub(r"\s+", " ", text)
    return text[:150]

# ==============================================================================
# 2. Advanced Semantic Analysis (Logic เดิม - ห้ามตัด)
# ==============================================================================
_INTENT_KEYWORDS = {
    "installation": ["install", "setup", "mounting", "connection", "wiring", "การติดตั้ง", "ต่อสาย"],
    "operation": ["operate", "use", "function", "start", "stop", "การใช้งาน", "วิธีใช้"],
    "troubleshooting": ["error", "fault", "problem", "solution", "fix", "แก้ปัญหา", "อาการเสีย"],
    "maintenance": ["maintain", "clean", "replace", "check", "inspection", "บำรุงรักษา"],
    "safety": ["safety", "warning", "caution", "danger", "hazard", "ความปลอดภัย", "อันตราย"],
    "specification": ["spec", "dimension", "weight", "voltage", "technical data", "ข้อมูลจำเพาะ"]
}

_ENTITY_KEYWORDS = [
    "power button", "led", "lcd", "battery", "fuse", "sensor", "switch", 
    "terminal", "cable", "motor", "pump", "valve", "controller"
]

def _detect_block_type(text: str) -> str:
    text_upper = text.upper()
    if re.match(r"^(WARNING|CAUTION|DANGER|คำเตือน|ข้อควรระวัง)[:\s]", text_upper):
        return "warning"
    if re.match(r"^(NOTE|NOTICE|IMPORTANT|หมายเหตุ|สำคัญ|ข้อสังเกต)[:\s]", text_upper):
        return "note"
    if re.match(r"^(\d+\.|Step\s+\d+|ขั้นตอนที่\s+\d+|[A-Z]\.)\s", text, re.IGNORECASE):
        return "step"
    return "normal"

def _analyze_intent(text: str, section: str) -> List[str]:
    combined = (text + " " + (section or "")).lower()
    intents = []
    for intent, keywords in _INTENT_KEYWORDS.items():
        if any(k in combined for k in keywords):
            intents.append(intent)
    return list(set(intents))

def _extract_entities(text: str) -> List[str]:
    text_lower = text.lower()
    found = []
    for entity in _ENTITY_KEYWORDS:
        if entity in text_lower:
            found.append(entity)
    return found

def _determine_answer_scope(block_type: str) -> str:
    if block_type == "step": return "procedure"
    if block_type == "warning": return "warning"
    if block_type == "note": return "note"
    return "general"

# ==============================================================================
# 3. Layout Analysis & Sorting (Logic เดิม - ห้ามตัด)
# ==============================================================================
def _detect_header_footer(blocks: List[dict], page_height: float) -> List[dict]:
    # Heuristic: บนสุด 7% คือ Header, ล่างสุด 7% คือ Footer
    HEADER_THRESH = page_height * 0.07
    FOOTER_THRESH = page_height * 0.93
    
    for b in blocks:
        x0, y0, x1, y1 = b.get("bbox", (0,0,0,0))
        if "extra" not in b: b["extra"] = {}
        
        is_header = y1 < HEADER_THRESH
        is_footer = y0 > FOOTER_THRESH
        
        b["extra"]["is_header"] = is_header
        b["extra"]["is_footer"] = is_footer
        
        # Mark as noise initially (can be reclaimed later if needed)
        if is_header or is_footer:
            b["extra"]["noise"] = True
            
    return blocks

def _sort_blocks_reading_order(blocks: List[dict]) -> List[dict]:
    # เรียงบรรทัด (Y) แล้วค่อยเรียงคอลัมน์ (X)
    # ใช้ Grid 12px เพื่อ group บรรทัดที่ใกล้กัน
    return sorted(blocks, key=lambda b: (int(b["bbox"][1] / 12), b["bbox"][0]))

# ==============================================================================
# 4. Smart Merge Logic (Logic เดิม - ห้ามตัด)
# ==============================================================================
def _merge_text_blocks(blocks: List[TextBlock]) -> List[TextBlock]:
    """รวม Block ที่ควรอยู่ด้วยกัน (เช่น Paragraph เดียวกัน หรือ Step เดียวกัน)"""
    if not blocks: return []
    
    merged: List[TextBlock] = []
    current = blocks[0]
    
    for next_block in blocks[1:]:
        curr_type = current.extra.get("block_type", "normal")
        next_type = next_block.extra.get("block_type", "normal")
        
        # ระยะห่างแนวตั้ง
        vertical_dist = next_block.bbox[1] - current.bbox[3]
        
        # Check Step Sequence (1. ... -> 2. ...)
        is_step_sequence = (curr_type in ["step", "step_sequence"] and next_type == "step")
        
        # Check Paragraph Merge (Font ใกล้กัน, ระยะห่างน้อย)
        curr_font = current.extra.get("font_size", 0)
        next_font = next_block.extra.get("font_size", 0)
        font_diff = abs(curr_font - next_font)
        
        is_paragraph = (
            curr_type == "normal" and next_type == "normal" and
            font_diff < 1.5 and 
            vertical_dist < 15.0 and 
            vertical_dist > -5.0 # ยอมให้ overlap นิดหน่อย
        )
        
        same_section = (current.section == next_block.section)

        if same_section and (is_step_sequence or is_paragraph):
            # MERGE
            delimiter = "\n" if is_step_sequence else " "
            current.content += delimiter + next_block.content
            
            # Update BBox
            current.bbox = (
                min(current.bbox[0], next_block.bbox[0]),
                min(current.bbox[1], next_block.bbox[1]),
                max(current.bbox[2], next_block.bbox[2]),
                max(current.bbox[3], next_block.bbox[3]),
            )
            
            if is_step_sequence:
                current.extra["block_type"] = "step_sequence"
                current.extra["answer_scope"] = "procedure"
            
            # Merge Metadata
            curr_intents = set(current.extra.get("intent", []))
            next_intents = set(next_block.extra.get("intent", []))
            current.extra["intent"] = list(curr_intents.union(next_intents))
            
            curr_entities = set(current.extra.get("entities", []))
            next_entities = set(next_block.extra.get("entities", []))
            current.extra["entities"] = list(curr_entities.union(next_entities))
                
        else:
            # PUSH & RESET
            merged.append(current)
            current = next_block
            
    merged.append(current)
    return merged

# ==============================================================================
# 5. Page Extraction Logic (Logic เดิม - ห้ามตัด)
# ==============================================================================
def _extract_text_blocks_from_page(
    pdf_page: fitz.Page,
    doc_id: str,
    page_number: int,
    start_index: int = 0,
    current_section: Optional[str] = None
) -> Tuple[List[TextBlock], Optional[str]]:
    
    try:
        # ใช้ dict เพื่อเอา metadata ละเอียด
        page_dict = pdf_page.get_text("dict", flags=fitz.TEXT_PRESERVE_IMAGES)
    except Exception as e:
        logger.warning(f"Page {page_number} dict extraction failed: {e}")
        # Fallback to plain text
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

    # Processing Blocks
    raw_blocks = page_dict.get("blocks", []) or []
    raw_blocks = _detect_header_footer(raw_blocks, pdf_page.rect.height)
    raw_blocks = _sort_blocks_reading_order(raw_blocks)

    # Calculate Page Median Font (for Header Detection)
    font_sizes = []
    for b in raw_blocks:
        if b.get("type") != 0: continue # Skip image blocks
        for line in b.get("lines", []):
            for span in line.get("spans", []):
                if span.get("size"): font_sizes.append(span["size"])
    
    page_median_font = statistics.median(font_sizes) if font_sizes else 10.0

    temp_blocks: List[TextBlock] = []
    current_index = start_index
    active_section = current_section
    
    for block in raw_blocks:
        if block.get("type") != 0: continue # Skip images here (handled by image_extractor)

        # Reconstruct Text
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
        
        # Filtering
        if not _is_meaningful_text(content): continue
        if block.get("extra", {}).get("noise", False): continue

        # Header Detection Logic
        avg_font = sum(block_fonts)/len(block_fonts) if block_fonts else 0
        is_heading = False
        heading_level = None
        
        if avg_font > page_median_font * 1.2 and len(content) < 200:
            if not re.match(r"^[\d\.\,\s]+$", content): # ไม่ใช่ตัวเลขล้วน
                is_heading = True
                heading_level = "H1" if avg_font > page_median_font * 1.5 else "H2"

        # Semantic Tagging
        block_type = "heading" if is_heading else _detect_block_type(content)
        
        # Propagate Section
        if is_heading:
            normalized_header = _normalize_section_title(content)
            if normalized_header: active_section = normalized_header

        # Metadata Enrichment
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
                "intent": intents,
                "answer_scope": answer_scope,
                "entities": entities
            }
        )
        temp_blocks.append(tb)

    # Smart Merge
    merged_blocks = _merge_text_blocks(temp_blocks)
    return merged_blocks, active_section


# ==============================================================================
# 6. Main Parse Function (Integrated)
# ==============================================================================
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

        # -----------------------------------------------------------
        # Step 1: Extract Text (Page by Page)
        # -----------------------------------------------------------
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
        # Step 2: Extract Tables & Embed
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
                vs = get_vector_store()
                for table in extracted_tables:
                    # [FIX] เรียกใช้ table_to_text ที่นำเข้ามาแล้ว
                    text = table_to_text(table)
                    metadata_dict = {
                        "doc_id": table.doc_id,
                        "page": table.page,
                        "source": "table",
                        "category": table.category,
                        "table_id": table.id,
                        "doc_type": doc_type
                    }
                    vs.add_texts(
                        texts=[text],
                        metadatas=[metadata_dict],
                        ids=[f"{table.doc_id}_{table.id}"]
                    )
            except Exception as e:
                logger.error(f"[pdf_parser] Failed to embed tables: {e}")

        # -----------------------------------------------------------
        # Step 3: Extract Images (New Integration)
        # -----------------------------------------------------------
        logger.info(f"[pdf_parser] Extracting images for {doc_id}...")
        ingested_dir = Path("ingested")
        
        try:
            extracted_images = extract_images(
                file_path=path,
                doc_id=doc_id,
                output_root=ingested_dir
            )
        except Exception as e:
            logger.error(f"[pdf_parser] Image extraction failed: {e}")
            extracted_images = []

        # -----------------------------------------------------------
        # Step 4: Construct Final Document
        # -----------------------------------------------------------
        final_doc = IngestedDocument(
            metadata=metadata,
            texts=all_text_blocks,
            tables=extracted_tables,
            images=extracted_images, 
        )

        # -----------------------------------------------------------
        # Step 5: Generate Smart Markdown Report (New Integration)
        # -----------------------------------------------------------
        doc_output_dir = ingested_dir / doc_id
        doc_output_dir.mkdir(parents=True, exist_ok=True)
        try:
            generate_markdown(final_doc, doc_output_dir)
        except Exception as e:
            logger.error(f"[pdf_parser] Markdown generation failed: {e}")

        return final_doc

    finally:
        pdf_doc.close()

# ==============================================================================
# CLI Test
# ==============================================================================
if __name__ == "__main__":
    import json
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("pdf_path")
    args = parser.parse_args()

    try:
        doc = parse_pdf(args.pdf_path)
        print(f"✅ Processed {len(doc.texts)} texts, {len(doc.tables)} tables, {len(doc.images)} images.")
    except Exception as e:
        print(f"❌ Error: {e}")