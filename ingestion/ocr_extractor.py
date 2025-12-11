from __future__ import annotations

import io
import re
import time
import json
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Set

import fitz  # PyMuPDF
import requests
import urllib3

from ingestion.config import OCR_API_URL, OCR_USERNAME, OCR_PASSWORD, VERIFY_SSL

# ปิด Warning กรณี SSL verify=False
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

_CACHED_TOKEN = None
_TOKEN_EXPIRY = 0

def _get_api_token() -> str:
    global _CACHED_TOKEN, _TOKEN_EXPIRY
    if _CACHED_TOKEN and time.time() < _TOKEN_EXPIRY - 60:
        return _CACHED_TOKEN

    login_url = f"{OCR_API_URL}/login"
    payload = {
        "username": OCR_USERNAME,
        "password": OCR_PASSWORD
    }
    
    try:
        response = requests.post(login_url, data=payload, verify=VERIFY_SSL, timeout=10)
        response.raise_for_status()
        data = response.json()
        token = data.get("access_token")
        if not token:
            raise ValueError("No access_token in login response")
        _CACHED_TOKEN = token
        _TOKEN_EXPIRY = time.time() + (30 * 60) 
        return token
    except Exception as e:
        print(f"[OCR-API] Login Failed: {e}")
        raise

def pdf_page_to_image_bytes(page: fitz.Page, dpi: int = 200) -> bytes:
    # ใช้ DPI 200 ตามที่เทสแล้วผ่าน
    pix = page.get_pixmap(dpi=dpi)
    return pix.tobytes("png")

def _clean_text(text: str) -> str:
    if not text: return ""
    # ลบตัวอักษรขยะที่มองไม่เห็น แต่เก็บ newline ไว้
    text = "".join(ch for ch in text if ch == "\n" or ch.isprintable())
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r" *\n *", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()

def ocr_page_via_api(image_bytes: bytes) -> str:
    try:
        token = _get_api_token()
        url = f"{OCR_API_URL}/process-file"
        headers = {"Authorization": f"Bearer {token}"}
        files = {'file': ('page.png', image_bytes, 'image/png')}
        data = {"ocr_engine": "tesseract", "lang": "th+eng"}

        response = requests.post(url, headers=headers, files=files, data=data, verify=VERIFY_SSL, timeout=45)
        
        if response.status_code != 200:
            print(f"❌ [DEBUG] API Error {response.status_code}: {response.text}")
            return ""
            
        result_json = response.json()
        
        # --- ส่วนที่แก้ไข: รองรับ Structure แบบซ้อนชั้น ---
        text = ""
        
        # 1. ลองดึงแบบซ้อนชั้น (extracted_text -> pages -> content)
        if "extracted_text" in result_json:
            pages = result_json["extracted_text"].get("pages", [])
            text_parts = []
            for p in pages:
                if "content" in p:
                    text_parts.append(p["content"])
            text = "\n".join(text_parts)
            
        # 2. ถ้าไม่เจอ ให้ลองดึงแบบชั้นเดียว (เผื่อ API เปลี่ยน format)
        if not text:
            text = result_json.get("text") or result_json.get("result") or result_json.get("content") or ""

        # Debug: ถ้ายังไม่เจออีก ให้ปริ้นท์มาดู
        if not text:
            print(f"⚠️ [DEBUG] Text extraction failed. JSON Structure: {list(result_json.keys())}")

        return _clean_text(str(text))

    except Exception as e:
        print(f"❌ [OCR-API] Exception: {e!r}")
        return ""

@dataclass
class OCRDocument:
    texts: List[Dict[str, Any]] = field(default_factory=list)

def ocr_extract_document(pdf_path: str, target_pages: Optional[Set[int]] = None) -> OCRDocument:
    doc = fitz.open(pdf_path)
    result = OCRDocument()
    
    if not target_pages:
        doc.close()
        return result

    print(f"[OCR] Processing {len(target_pages)} specific pages...")

    for idx, page in enumerate(doc):
        page_no = idx + 1
        if page_no in target_pages:
            print(f"[OCR] Scanning Page {page_no}...", end=" ", flush=True)
            
            image_bytes = pdf_page_to_image_bytes(page, dpi=200)
            ocr_text = ocr_page_via_api(image_bytes)
            
            if ocr_text:
                print(f"✅ Got {len(ocr_text)} chars.")
                result.texts.append({
                    "page": page_no,
                    "content": ocr_text,
                    "source": "ocr_api_tesseract"
                })
            else:
                print("❌ No text captured.")
    
    doc.close()
    return result