from __future__ import annotations

import io
import re
import time
import json
import os # [NEW]
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Set

import fitz  # PyMuPDF
import requests
import urllib3
import cv2
import numpy as np

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

def pdf_page_to_image_bytes(page: fitz.Page, dpi: int = 300) -> bytes:
    pix = page.get_pixmap(dpi=dpi)
    return pix.tobytes("png")

def _clean_text(text: str) -> str:
    if not text: return ""
    text = "".join(ch for ch in text if ch == "\n" or ch.isprintable())
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r" *\n *", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()

def _has_meaningful_text(text: str) -> bool:
    if not text: return False
    matches = _WORD_CHARS_PATTERN.findall(text)
    return len(matches) > 5

def _preprocess_image_cv2(image_bytes: bytes, debug_name: str = None) -> bytes:
    """
    เตรียมภาพสำหรับ Tesseract: Grayscale -> Denoise -> Thresholding (Otsu's)
    """
    if not image_bytes: return b""
    nparr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img is None: return image_bytes

    # 1. Grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 2. Denoise
    gray = cv2.medianBlur(gray, 3)

    # 3. Thresholding (Otsu)
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # [NEW] Save Debug Image ถ้าต้องการ
    if debug_name:
        cv2.imwrite(debug_name, binary)
        print(f"   [DEBUG] Saved processed image to: {debug_name}")

    _, encoded_img = cv2.imencode('.png', binary)
    return encoded_img.tobytes()

def _send_to_ocr_api(image_bytes: bytes) -> str:
    """ฟังก์ชันย่อยสำหรับส่ง Request จริงๆ"""
    try:
        token = _get_api_token()
        url = f"{OCR_API_URL}/process-file"
        headers = {"Authorization": f"Bearer {token}"}
        
        files = {'file': ('page.png', image_bytes, 'image/png')}
        data = {
            "ocr_engine": "tesseract", 
            "lang": "tha+eng", # ลองใช้ tha (3-letter code) เผื่อ API รองรับ
            "options": "--psm 6" # ลองเพิ่ม PSM 6 (Assume single block) ถ้า API รองรับ
        }

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

def ocr_page_via_api(image_bytes: bytes) -> str:
    # 1. ลองแบบ Preprocess (High Contrast) ก่อน
    print("      > Trying Method 1: Preprocessed (B&W)...")
    processed_bytes = _preprocess_image_cv2(image_bytes, debug_name="debug_ocr_processed.png")
    text_v1 = _send_to_ocr_api(processed_bytes)
    
    # 2. Smart Retry: ถ้าได้ข้อความน้อยเกินไป (< 100 ตัว) ให้ลองส่งภาพ Original (Grayscale)
    if len(text_v1) < 100:
        print(f"      > Result too short ({len(text_v1)} chars). Retrying with Original Image...")
        
        # Save Debug Original
        with open("debug_ocr_original.png", "wb") as f:
            f.write(image_bytes)

        text_v2 = _send_to_ocr_api(image_bytes)
        
        print(f"      > Method 2 Result: {len(text_v2)} chars.")
        
        # เลือกอันที่ยาวกว่า
        if len(text_v2) > len(text_v1):
            return text_v2
            
    return text_v1

@dataclass
class OCRDocument:
    texts: List[Dict[str, Any]] = field(default_factory=list)

def ocr_extract_document(pdf_path: str, target_pages: Optional[Set[int]] = None) -> OCRDocument:
    doc = fitz.open(pdf_path)
    result = OCRDocument()
    
    if target_pages is None:
        print("[OCR] Checking for existing text layer...")
        target_pages = set()
        for idx, page in enumerate(doc):
            raw_text = _clean_text(page.get_text("text") or "")
            if _has_meaningful_text(raw_text):
                print(f"   ✅ Page {idx+1}: Found digital text ({len(raw_text)} chars). Using it.")
                result.texts.append({
                    "page": idx + 1,
                    "content": raw_text,
                    "source": "pdf_text"
                })
            else:
                print(f"   ⚠️ Page {idx+1}: No text found. Marking for OCR.")
                target_pages.add(idx + 1)
        
        if not target_pages:
            result.texts.sort(key=lambda x: x["page"])
            doc.close()
            return result

    if target_pages:
        print(f"[OCR] Sending {len(target_pages)} image-based pages to API...")
        for idx, page in enumerate(doc):
            page_no = idx + 1
            if page_no in target_pages:
                print(f"   - OCR Scanning Page {page_no}...", end=" ", flush=True)
                image_bytes = pdf_page_to_image_bytes(page)
                ocr_text = ocr_page_via_api(image_bytes)
                
                if ocr_text:
                    print(f"✅ Final Result: {len(ocr_text)} chars.")
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