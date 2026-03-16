# ingestion/document_classifier.py
import litellm
import logging
from .config import CUSTOM_API_BASE, CUSTOM_API_KEY, CUSTOM_MODEL_NAME

logger = logging.getLogger(__name__)

class DocumentClassifier:
    def __init__(self):
        self.model = f"openai/{CUSTOM_MODEL_NAME}"
        self.api_base = CUSTOM_API_BASE
        self.api_key = CUSTOM_API_KEY

    def classify(self, text_preview: str) -> str:
        if not text_preview or len(text_preview.strip()) < 10:
            return "generic_doc"

        prompt = f"วิเคราะห์ประเภทเอกสารเพียงคำเดียว (statement, invoice, manual, contract, generic_doc) จากเนื้อหานี้: {text_preview[:1000]}"
        try:
            response = litellm.completion(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                api_base=self.api_base,
                api_key=self.api_key,
                temperature=0.1
            )
            return response.choices[0].message.content.strip().lower()
        except Exception as e:
            logger.error(f"Classifier Error: {e}")
            return "generic_doc"

# [IMPORTANT] เพิ่มฟังก์ชันนี้เพื่อให้ scripts/run_ingestion.py เรียกใช้ได้
def classify_document(text_preview: str) -> str:
    classifier = DocumentClassifier()
    return classifier.classify(text_preview)