import re
import litellm
import logging
from typing import Dict, Any, List
from .config import CUSTOM_API_BASE, CUSTOM_API_KEY, CUSTOM_MODEL_NAME
from .schema import TextBlock, TableBlock

logger = logging.getLogger(__name__)

class DataCleaner:
    def __init__(self):
        self.model = f"openai/{CUSTOM_MODEL_NAME}"
        self.api_base = CUSTOM_API_BASE
        self.api_key = CUSTOM_API_KEY

    def normalize_text(self, text: str) -> str:
        """ลบช่องว่างส่วนเกินและจัดการสระไทยที่เพี้ยน"""
        if not text: return ""
        text = re.sub(r'(?<=[฀-๿])\s+(?=[฀-๿])', '', text)
        text = re.sub(r'[ \t]+', ' ', text)
        text = re.sub(r'\n\s*\n', '\n\n', text)
        return text.strip()

    def clean_text_with_ai(self, raw_text: str) -> str:
        """ใช้ Qwen เกลาคำผิดแทน Gemini เพื่อแก้ Error 404"""
        if not raw_text or len(raw_text.strip()) < 20:
            return raw_text
        try:
            print(f"   🤖 [Qwen-Corrector] Refining text...")
            response = litellm.completion(
                model=self.model,
                messages=[
                    {"role": "system", "content": "แก้ไขคำสะกดผิดจากการ OCR ให้ถูกต้องตามหลักภาษาไทย โดยห้ามสรุปความ"},
                    {"role": "user", "content": raw_text}
                ],
                api_base=self.api_base,
                api_key=self.api_key,
                temperature=0.0
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            logger.warning(f"AI Correction failed: {e}")
            return raw_text

    def process_document_cleaning(self, doc_data: Dict[str, Any]) -> Dict[str, Any]:
        """วนลูปจัดการ Cleaning ทุกหน้าในเอกสาร"""
        if "texts" not in doc_data: return doc_data
        for page in doc_data["texts"]:
            base_clean = self.normalize_text(page.get("content", ""))
            page["content"] = self.clean_text_with_ai(base_clean)
            page["is_cleaned"] = True
        return doc_data


# ============================================================
# ฟังก์ชันระดับ module ที่ scripts/run_cleaning.py ใช้งาน
# ============================================================

def _clean_cell(value: Any) -> Any:
    """ทำความสะอาดข้อมูลใน cell ของตาราง (รักษาชนิดของตัวเลข)"""
    if value is None:
        return value
    if isinstance(value, (int, float, bool)):
        return value
    text = str(value)
    text = re.sub(r'(?<=[฀-๿])\s+(?=[฀-๿])', '', text)
    text = re.sub(r'[ \t]+', ' ', text)
    text = text.replace('\x00', '')
    return text.strip()


def clean_text_blocks(blocks: List[TextBlock]) -> List[TextBlock]:
    """
    ทำความสะอาด TextBlock:
    - Normalize ช่องว่าง / สระไทยลอย
    - รักษา metadata เดิมทั้งหมด (id, doc_id, page, bbox, extra, ฯลฯ)
    - ไม่เรียก LLM เพื่อให้ทำงานเร็ว (ถ้าอยากใช้ LLM ให้ใช้ DataCleaner.clean_text_with_ai)
    """
    cleaner = DataCleaner()
    cleaned: List[TextBlock] = []
    for b in blocks:
        new_content = cleaner.normalize_text(b.content or "")
        if not new_content:
            continue
        extra = dict(b.extra or {})
        extra["is_cleaned"] = True
        cleaned.append(
            TextBlock(
                id=b.id,
                doc_id=b.doc_id,
                page=b.page,
                content=new_content,
                section=b.section,
                category=b.category,
                role=b.role,
                bbox=b.bbox,
                extra=extra,
            )
        )
    return cleaned


def clean_table_blocks(tables: List[TableBlock]) -> List[TableBlock]:
    """
    ทำความสะอาด TableBlock:
    - Normalize header และ cell content (ตัดช่องว่างส่วนเกิน)
    - รักษาโครงสร้าง rows/columns เดิม
    - รักษา metadata เดิมทั้งหมด
    """
    cleaned: List[TableBlock] = []
    for tb in tables:
        columns = [str(_clean_cell(c) or "") for c in (tb.columns or [])]
        rows = [[_clean_cell(cell) for cell in row] for row in (tb.rows or [])]

        extra = dict(tb.extra or {})
        extra["is_cleaned"] = True

        cleaned.append(
            TableBlock(
                id=tb.id,
                doc_id=tb.doc_id,
                page=tb.page,
                name=tb.name,
                section=tb.section,
                category=tb.category,
                role=tb.role,
                columns=columns,
                rows=rows,
                markdown=tb.markdown,
                html_content=tb.html_content,
                image_path=tb.image_path,
                is_complex=tb.is_complex,
                source=tb.source,
                method=tb.method,
                numeric_trust=tb.numeric_trust,
                structured_available=tb.structured_available,
                raw_available=tb.raw_available,
                structure_lossy=tb.structure_lossy,
                bbox=tb.bbox,
                extra=extra,
            )
        )
    return cleaned
