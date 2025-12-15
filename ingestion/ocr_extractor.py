from __future__ import annotations

import io
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional

import fitz  # PyMuPDF
from PIL import Image
import google.generativeai as genai

from ingestion.config import GOOGLE_API_KEY


# -------------------------------------------------------------------
# Helper: OCR client / model
# -------------------------------------------------------------------


def _get_gemini_model():
    """
    เตรียม client + model ของ Gemini สำหรับ OCR

    ใช้ GOOGLE_API_KEY จาก ingestion.config
    """
    if not GOOGLE_API_KEY:
        raise ValueError("❌ Missing GOOGLE_API_KEY in config.py")

    # client-style API (เวอร์ชันที่โปรเจกต์ใช้อยู่)
    client = genai.Client(api_key=GOOGLE_API_KEY)

    # ถ้าอนาคตเปลี่ยนชื่อโมเดล ค่อยมาแก้จุดนี้จุดเดียว
    model = client.models.get("gemini-2.5-flash")
    return client, model


# -------------------------------------------------------------------
# Helper: PDF page → image bytes
# -------------------------------------------------------------------


def pdf_page_to_image_bytes(page: fitz.Page, dpi: int = 200) -> bytes:
    """
    แปลงหนึ่งหน้า PDF เป็น PNG bytes สำหรับส่งให้ Gemini OCR
    """
    pix = page.get_pixmap(dpi=dpi)
    return pix.tobytes("png")


# -------------------------------------------------------------------
# Helper: text cleaning / heuristic
# -------------------------------------------------------------------

import re

_WORD_CHARS_PATTERN = re.compile(r"[A-Za-z0-9\u0E00-\u0E7F]")


def _clean_text(text: str) -> str:
    """
    ทำความสะอาดข้อความ OCR:
    - ลบ control chars แปลก ๆ
    - แทน multiple spaces / tabs เป็น space เดียว
    - เก็บ newline ไว้พอประมาณ
    """
    if not text:
        return ""

    # ลบ control characters (ยกเว้น newline)
    text = "".join(ch for ch in text if ch == "\n" or ch.isprintable())

    # ลบ prefix/suffix ที่โมเดลชอบใส่มาเอง เช่น "Here is the text:" (กันไว้)
    # ไม่ strict มาก แค่กันเคสหลุด ๆ
    text = re.sub(r"^\s*(OCR\s+Result:|Here is.+?:)\s*", "", text, flags=re.IGNORECASE)

    # แทนหลาย space/tab ด้วย space เดียว
    text = re.sub(r"[ \t]+", " ", text)

    # ลบ space รอบ newline
    text = re.sub(r" *\n *", "\n", text)

    # ตัด newline ซ้อนให้เหลือไม่เกิน 2 บรรทัดติดกัน
    text = re.sub(r"\n{3,}", "\n\n", text)

    return text.strip()


def _has_enough_text(text: str, min_chars: int = 10) -> bool:
    """
    เช็กว่าข้อความมีเนื้อหาพอสมควรไหม:
    - ต้องมีตัวอักษร/ตัวเลข/ตัวไทยอย่างน้อย min_chars ตัว
    """
    if not text:
        return False

    matches = _WORD_CHARS_PATTERN.findall(text)
    return len(matches) >= min_chars


# -------------------------------------------------------------------
# OCR call
# -------------------------------------------------------------------


def ocr_page(client, model, image_bytes: bytes) -> str:
    """
    เรียก Gemini ทำ OCR จาก image bytes
    คืนค่าเป็นข้อความที่ผ่านการ clean แล้ว
    """
    # ใส่ prompt บอกให้ส่งเฉพาะเนื้อหาในเอกสาร ไม่ต้องอธิบาย
    try:
        response = client.models.generate_content(
            model=model.name,
            contents=[
                {
                    "mime_type": "image/png",
                    "data": image_bytes,
                },
                {
                    "role": "user",
                    "parts": [
                        {
                            "text": (
                                "โปรดอ่านข้อความทั้งหมดที่อยู่ในภาพนี้แล้วส่งคืนเป็นข้อความธรรมดา "
                                "รักษาบรรทัดและโครงสร้างเท่าที่จำเป็น "
                                "ห้ามเพิ่มคำอธิบายอื่น ห้ามใส่คำว่า 'OCR Result' หรือข้อความประกอบใด ๆ เพิ่มเติม"
                            )
                        }
                    ],
                },
            ],
        )
    except Exception as e:  # noqa: BLE001
        print(f"[OCR] Gemini OCR error: {e!r}")
        return ""

    text = getattr(response, "text", "") or ""
    return _clean_text(text)


# -------------------------------------------------------------------
# OCRDocument dataclass
# -------------------------------------------------------------------


@dataclass
class OCRDocument:
    """
    เก็บผล OCR ทั้งเอกสาร
    texts: list ของ dict {"page": int, "content": str}
    """

    texts: List[Dict[str, Any]] = field(default_factory=list)


# -------------------------------------------------------------------
# main function
# -------------------------------------------------------------------


def ocr_extract_document(pdf_path: str) -> OCRDocument:
    """
    OCR ทั้งเอกสาร PDF แบบฉลาดขึ้น:

    - ถ้าหน้ามี text อยู่แล้วจาก PDF (page.get_text("text"))
      และมีตัวอักษรพอสมควร → ใช้ text นั้นเลย (ไม่ยิง Gemini)
    - ถ้าหน้าว่าง/เป็นรูป/ตัวอักษรเละ → แปลงหน้าเป็นภาพ แล้วส่งให้ Gemini OCR
    """
    client, model = _get_gemini_model()

    doc = fitz.open(pdf_path)
    result = OCRDocument()

    total_pages = len(doc)
    print(f"[OCR] Total pages: {total_pages}")

    for idx, page in enumerate(doc):
        page_no = idx + 1
        print(f"[OCR] Processing page {page_no}/{total_pages}")

        # 1) ลองดึง text ปกติจาก PDF ก่อน
        try:
            pdf_text = page.get_text("text") or ""
        except Exception as e:  # noqa: BLE001
            print(f"[OCR] page.get_text('text') error on page {page_no}: {e!r}")
            pdf_text = ""

        pdf_text_clean = _clean_text(pdf_text)

        if _has_enough_text(pdf_text_clean, min_chars=10):
            # หน้านี้เป็น text PDF อยู่แล้ว → ไม่ต้องเปลือง OCR
            content = pdf_text_clean
            source = "pdf_text"
        else:
            # หน้านี้น่าจะเป็นสแกน/ภาพ → ใช้ OCR
            image_bytes = pdf_page_to_image_bytes(page, dpi=200)
            ocr_text = ocr_page(client, model, image_bytes)
            if not _has_enough_text(ocr_text, min_chars=5):
                print(f"[OCR] Warning: OCR result on page {page_no} is very short/empty.")
            content = ocr_text
            source = "ocr"

        result.texts.append(
            {
                "page": page_no,
                "content": content,
                # ไม่กระทบโค้ดเดิม แต่ถ้าอยาก debug ก็รู้ได้ว่ามาจากไหน
                "source": source,
            }
        )

    doc.close()
    print("[OCR] Completed")
    return result
