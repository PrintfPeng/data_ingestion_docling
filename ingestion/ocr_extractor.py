# ingestion/ocr_extractor.py
import time
import fitz
import requests
import litellm
import urllib3
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Set
from ingestion.config import (
    OCR_API_URL, OCR_USERNAME, OCR_PASSWORD, VERIFY_SSL,
    CUSTOM_API_BASE, CUSTOM_API_KEY, CUSTOM_MODEL_NAME
)

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

_CACHED_TOKEN = None
_TOKEN_EXPIRY = 0

def _get_api_token() -> str:
    global _CACHED_TOKEN, _TOKEN_EXPIRY
    if _CACHED_TOKEN and time.time() < _TOKEN_EXPIRY - 60:
        return _CACHED_TOKEN
    try:
        response = requests.post(f"{OCR_API_URL}/login", 
                                 data={"username": OCR_USERNAME, "password": OCR_PASSWORD}, 
                                 verify=VERIFY_SSL, timeout=10)
        token = response.json().get("access_token")
        _CACHED_TOKEN = token
        _TOKEN_EXPIRY = time.time() + (30 * 60)
        return token
    except Exception as e:
        print(f"Login Failed: {e}")
        raise

def refine_text(text: str) -> str:
    if not text: return ""
    try:
        response = litellm.completion(
            model=f"openai/{CUSTOM_MODEL_NAME}",
            messages=[{"role": "user", "content": f"แก้ไขคำผิด OCR ภาษาไทย: {text}"}],
            api_base=CUSTOM_API_BASE,
            api_key=CUSTOM_API_KEY
        )
        return response.choices[0].message.content.strip()
    except: return text

def ocr_page_via_api(image_bytes: bytes) -> str:
    token = _get_api_token()
    files = {'file': ('page.png', image_bytes, 'image/png')}
    data = {"ocr_engine": "tesseract", "lang": "tha+eng", "options": "--psm 3"}
    try:
        res = requests.post(f"{OCR_API_URL}/process-file", 
                            headers={"Authorization": f"Bearer {token}"}, 
                            files=files, data=data, verify=VERIFY_SSL, timeout=45)
        raw_text = res.json().get("extracted_text", {}).get("pages", [{}])[0].get("content", "")
        return refine_text(raw_text)
    except: return ""

@dataclass
class OCRDocument:
    texts: List[Dict[str, Any]] = field(default_factory=list)

def ocr_extract_document(pdf_path: str, target_pages: Optional[Set[int]] = None) -> OCRDocument:
    doc = fitz.open(pdf_path)
    result = OCRDocument()
    pages_to_scan = target_pages if target_pages else range(1, len(doc) + 1)
    for p_no in pages_to_scan:
        page = doc[p_no - 1]
        text = ocr_page_via_api(page.get_pixmap(dpi=300).tobytes("png"))
        result.texts.append({"page": p_no, "content": text, "source": "qwen_ocr"})
    doc.close()
    return result