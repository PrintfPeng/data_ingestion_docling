import re
import litellm
import logging
from typing import Dict, Any, List
from .config import CUSTOM_API_BASE, CUSTOM_API_KEY, CUSTOM_MODEL_NAME

logger = logging.getLogger(__name__)

class DataCleaner:
    def __init__(self):
        self.model = f"openai/{CUSTOM_MODEL_NAME}"
        self.api_base = CUSTOM_API_BASE
        self.api_key = CUSTOM_API_KEY

    def normalize_text(self, text: str) -> str:
        """ลบช่องว่างส่วนเกินและจัดการสระไทยที่เพี้ยน"""
        if not text: return ""
        text = re.sub(r'(?<=[\u0E00-\u0E7F])\s+(?=[\u0E00-\u0E7F])', '', text)
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