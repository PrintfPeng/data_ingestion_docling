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
# เอา PIL ออกก็ได้ถ้าไม่ได้ใช้ preprocess แล้ว แต่ใส่ไว้เผื่อไฟล์อื่นที่จำเป็นต้อง OCR จริงๆ
from PIL import Image, ImageEnhance

from ingestion.config import OCR_API_URL, OCR_USERNAME, OCR_PASSWORD, VERIFY_SSL

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

_CACHED_TOKEN = None
_TOKEN_EXPIRY = 0
_WORD_CHARS_PATTERN = re.compile(r"[A-Za-z0-9\u0E00-\u0E7F]")

def _get_api_token() -> str:
    global _CACHED_TOKEN, _TOKEN_EXPIRY
    if _CACHED_TOKEN and time.time() < _TOKEN_EXPIRY - 60:
        return _CACHED_TOKEN

    login_url = f"{OCR_API_URL}/login"
    payload = {"username": OCR_USERNAME, "password": OCR_PASSWORD}
    
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

def pdf_page_to_image_bytes(page: fitz.Page, dpi: int = 200) -> bytes: # ลด DPI กลับมาปกติ
    pix = page.get_pixmap(dpi=dpi)
    return pix.tobytes("png")

def _clean_text(text: str) -> str:
    if not text: return ""
    # เก็บตัวอักษรไว้, ยุบ space, แต่รักษา newline
    text = "".join(ch for ch in text if ch == "\n" or ch.isprintable())
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r" *\n *", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()

def _has_meaningful_text(text: str) -> bool:
    """เช็คว่ามีตัวหนังสือจริงๆ ไหม (กันพวกมีแต่เลขหน้าหรือจุดไข่ปลา)"""
    if not text: return False
    # นับเฉพาะ ก-ฮ, A-Z, 0-9
    matches = _WORD_CHARS_PATTERN.findall(text)
    # ถ้ามีตัวอักษรเกิน 5 ตัว ถือว่าอ่านรู้เรื่องแล้ว (ไฟล์ของคุณมีเป็นร้อย)
    return len(matches) > 5

def ocr_page_via_api(image_bytes: bytes) -> str:
    # ... (Logic เดิม ไม่ต้อง Preprocess แล้วเพราะเราจะเลี่ยง OCR)
    try:
        token = _get_api_token()
        url = f"{OCR_API_URL}/process-file"
        headers = {"Authorization": f"Bearer {token}"}
        files = {'file': ('page.png', image_bytes, 'image/png')}
        data = {"ocr_engine": "tesseract", "lang": "th+eng"}

        response = requests.post(url, headers=headers, files=files, data=data, verify=VERIFY_SSL, timeout=45)
        if response.status_code != 200:
            return ""
            
        result_json = response.json()
        text = ""
        if "extracted_text" in result_json:
            pages = result_json["extracted_text"].get("pages", [])
            text_parts = []
            for p in pages:
                if "content" in p: text_parts.append(p["content"])
            text = "\n".join(text_parts)
            
        if not text:
            text = result_json.get("text") or result_json.get("result") or ""

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
    
    # 📌 1. ถ้าไม่ระบุหน้า -> ให้เช็คก่อนว่ามี Text Layer ไหม
    if target_pages is None:
        print("[OCR] Checking for existing text layer...")
        target_pages = set()
        
        for idx, page in enumerate(doc):
            # ดึง Text ดิบๆ จาก PDF
            raw_text = _clean_text(page.get_text("text") or "")
            
            # 📌 Logic สำคัญ: ถ้ามีตัวอักษรเกิน 5 ตัว ให้ใช้ Text นั้นเลย!
            if _has_meaningful_text(raw_text):
                print(f"   ✅ Page {idx+1}: Found digital text ({len(raw_text)} chars). Using it directly.")
                result.texts.append({
                    "page": idx + 1,
                    "content": raw_text,
                    "source": "pdf_text" # บอกว่าเป็น Text แท้ ไม่ใช่ OCR
                })
            else:
                # ถ้าหน้าขาวๆ หรือมีแต่รูป ค่อยส่ง OCR
                print(f"   ⚠️ Page {idx+1}: No text found. Marking for OCR.")
                target_pages.add(idx + 1)
        
        # ถ้าทุกหน้ามี Text หมดแล้ว ก็จบเลย ไม่ต้องยิง API
        if not target_pages:
            print("✨ All pages have text. Skipping OCR API completely.")
            result.texts.sort(key=lambda x: x["page"]) # เรียงหน้าให้สวยงาม
            doc.close()
            return result

    # 📌 2. เฉพาะหน้าที่ไม่มี Text จริงๆ ค่อยยิง API
    if target_pages:
        print(f"[OCR] Sending {len(target_pages)} image-based pages to API...")
        for idx, page in enumerate(doc):
            page_no = idx + 1
            if page_no in target_pages:
                print(f"   - OCR Scanning Page {page_no}...", end=" ", flush=True)
                image_bytes = pdf_page_to_image_bytes(page)
                ocr_text = ocr_page_via_api(image_bytes)
                if ocr_text:
                    print(f"✅ Got {len(ocr_text)} chars.")
                    result.texts.append({
                        "page": page_no,
                        "content": ocr_text,
                        "source": "ocr_api_tesseract"
                    })
                else:
                    print("❌ Failed.")

    result.texts.sort(key=lambda x: x["page"])
    doc.close()
    return result