# ingestion/document_classifier.py
import litellm
import logging
from .config import CUSTOM_API_BASE, CUSTOM_API_KEY, CUSTOM_MODEL_NAME
from .schema import IngestedDocument

logger = logging.getLogger(__name__)


class DocumentClassifier:
    def __init__(self):
        self.model = f"openai/{CUSTOM_MODEL_NAME}"
        self.api_base = CUSTOM_API_BASE
        self.api_key = CUSTOM_API_KEY

    def classify(self, text_preview: str) -> str:
        if not text_preview or len(text_preview.strip()) < 10:
            return "generic_doc"

        prompt = (
            f"วิเคราะห์ประเภทเอกสารเพียงคำเดียว "
            f"(statement, invoice, manual, contract, generic_doc) "
            f"จากเนื้อหานี้: {text_preview[:1000]}"
        )
        try:
            response = litellm.completion(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                api_base=self.api_base,
                api_key=self.api_key,
                temperature=0.1,
            )
            return response.choices[0].message.content.strip().lower()
        except Exception as e:
            logger.error(f"Classifier Error: {e}")
            return "generic_doc"


# ============================================================
# Rule-based Classifier (ไม่ต้องใช้ LLM)
# ============================================================
_RULE_KEYWORDS = {
    "bank_statement": [
        "รายการเดินบัญชี", "statement", "ยอดคงเหลือ", "เลขที่บัญชี", "account no",
        "withdrawal", "deposit", "balance",
    ],
    "invoice": [
        "invoice", "ใบแจ้งหนี้", "ใบวางบิล", "invoice no", "เลขที่ใบแจ้งหนี้",
    ],
    "receipt": [
        "receipt", "ใบเสร็จ", "ใบเสร็จรับเงิน", "receipt no",
    ],
    "contract": [
        "contract", "agreement", "สัญญา", "ข้อตกลง", "mou",
    ],
    "manual": [
        "manual", "handbook", "guide", "คู่มือ", "instruction",
    ],
    "qna": [
        "ถาม:", "ตอบ:", "คำถาม", "เฉลย", "question", "answer",
    ],
}


def _rule_based_classify(text_preview: str) -> str:
    lower = (text_preview or "").lower()
    scores = {}
    for label, keywords in _RULE_KEYWORDS.items():
        score = sum(1 for k in keywords if k in lower)
        if score > 0:
            scores[label] = score
    if not scores:
        return "generic"
    return max(scores.items(), key=lambda x: x[1])[0]


def classify_document(doc: IngestedDocument, use_llm: bool = False) -> str:
    """
    Classify document type จาก IngestedDocument
    - use_llm=False -> ใช้ rule-based keyword matching (default, เร็ว, ไม่ต้อง key)
    - use_llm=True  -> เรียก LLM ผ่าน DocumentClassifier
    ถูกเรียกโดย scripts/run_ingestion.py
    """
    # รวมเนื้อหาข้อความหน้าแรกๆ มาใช้เป็น preview
    preview_parts = []
    for t in (doc.texts or [])[:20]:
        content = (t.content or "").strip()
        if content:
            preview_parts.append(content)
        if sum(len(p) for p in preview_parts) > 2000:
            break
    text_preview = "\n".join(preview_parts)

    if use_llm:
        try:
            return DocumentClassifier().classify(text_preview)
        except Exception as e:
            logger.warning(f"LLM classify failed, fallback to rule-based: {e}")

    return _rule_based_classify(text_preview)
