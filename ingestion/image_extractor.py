from __future__ import annotations

"""
image_extractor.py

หน้าที่:
- เปิดไฟล์ PDF
- ดึงรูปภาพทุกภาพในเอกสาร
- บันทึกลงโฟลเดอร์ (เช่น ingested/{doc_id}/images/img_001_001.png)
- [NEW] ใช้ Gemini Vision (Flash) อ่านภาพและสร้าง Caption
- แปลงผลลัพธ์เป็น list[ImageBlock] ตาม schema
"""

import time
from pathlib import Path
from typing import List, Optional

import fitz  # PyMuPDF
import google.generativeai as genai
from PIL import Image

from .schema import ImageBlock
from .config import GOOGLE_API_KEY  # ดึง Key จาก config กลาง

# -------------------------------------------------------------------
# Helper: Gemini Vision
# -------------------------------------------------------------------

def _get_gemini_vision_model():
    """เตรียม Gemini 2.0 Flash สำหรับงาน Vision"""
    if not GOOGLE_API_KEY:
        print("[image_extractor] Warning: No GOOGLE_API_KEY. Image captioning will be skipped.")
        return None

    try:
        genai.configure(api_key=GOOGLE_API_KEY)
        # ใช้ Flash เพราะเร็วและถูก เหมาะกับงาน Caption จำนวนมาก
        return genai.GenerativeModel("gemini-2.0-flash")
    except Exception as e:
        print(f"[image_extractor] Failed to init Gemini: {e}")
        return None

def _generate_image_caption(model, image_path: Path) -> str:
    """ส่งรูปไปให้ AI อธิบาย (Captioning)"""
    if not model:
        return ""

    try:
        # เปิดรูปด้วย PIL
        img = Image.open(image_path)
        
        prompt = (
            "อธิบายรูปภาพนี้โดยละเอียด: "
            "1. ถ้าเป็นกราฟ/แผนภูมิ ให้บอกชื่อแกน ตัวเลขสำคัญ และแนวโน้ม "
            "2. ถ้าเป็นรูปถ่าย/ไดอะแกรม ให้บอกว่าคืออะไรและมีองค์ประกอบสำคัญอะไรบ้าง "
            "3. ถ้ามีข้อความในภาพ ให้อ่านและสรุปข้อความนั้นมาด้วย "
            "ตอบเป็นภาษาไทย กระชับและได้ใจความ"
        )

        response = model.generate_content([prompt, img])
        return response.text.strip()
    except Exception as e:
        print(f"[image_extractor] Caption generation failed for {image_path.name}: {e}")
        return ""

# -------------------------------------------------------------------
# Main Extraction
# -------------------------------------------------------------------

def extract_images(
    file_path: str | Path,
    doc_id: str,
    output_root: str | Path = "ingested",
) -> List[ImageBlock]:
    """
    ดึงรูปภาพทั้งหมดจาก PDF และสร้าง Caption ด้วย AI
    """
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"PDF file not found: {path}")

    output_root = Path(output_root)
    image_dir = output_root / doc_id / "images"
    image_dir.mkdir(parents=True, exist_ok=True)

    # Init AI Model
    vision_model = _get_gemini_vision_model()

    pdf_doc = fitz.open(path)
    image_blocks: List[ImageBlock] = []

    try:
        image_counter = 0

        # loop ทุกหน้า
        for page_index in range(pdf_doc.page_count):
            page = pdf_doc[page_index]
            page_number = page_index + 1

            # get_images(full=True) คืน list ของ image objects ในหน้านั้น
            images = page.get_images(full=True)

            for img_index, img in enumerate(images, start=1):
                xref = img[0]  # image reference id ใน PDF
                base_image = pdf_doc.extract_image(xref)

                img_bytes: bytes = base_image["image"]
                img_ext: str = base_image.get("ext", "png")
                width: int = base_image.get("width", 0)
                height: int = base_image.get("height", 0)

                # กรองรูปที่เล็กเกินไป (มักเป็น icon/line/noise)
                if width < 50 or height < 50:
                    continue

                image_counter += 1
                img_id = f"img_{image_counter:04d}"

                # ตั้งชื่อไฟล์ เช่น img_p001_001.png
                filename = f"img_p{page_number:03d}_{img_index:03d}.{img_ext}"
                file_path_on_disk = image_dir / filename

                # เซฟรูปลงดิสก์
                with open(file_path_on_disk, "wb") as f:
                    f.write(img_bytes)
                
                # [NEW] AI Captioning
                caption_text = ""
                if vision_model:
                    print(f"[image_extractor] Generating caption for {filename}...")
                    caption_text = _generate_image_caption(vision_model, file_path_on_disk)
                    # ใส่ delay นิดหน่อยกัน Rate Limit (ถ้าใช้ Free Tier)
                    time.sleep(1.0) 

                image_block = ImageBlock(
                    id=img_id,
                    doc_id=doc_id,
                    page=page_number,
                    file_path=str(file_path_on_disk),
                    caption=caption_text, # ใส่ผลลัพธ์จาก AI ลงไปตรงนี้!
                    section=None,
                    category=None,
                    bbox=None,
                    extra={
                        "width": width,
                        "height": height,
                        "xref": xref,
                        "ai_captioned": bool(caption_text)
                    },
                )
                image_blocks.append(image_block)

        return image_blocks

    finally:
        pdf_doc.close()


if __name__ == "__main__":
    import json
    import argparse

    parser = argparse.ArgumentParser(description="Extract images from PDF into ImageBlock list.")
    parser.add_argument("pdf_path", help="Path to PDF file")
    parser.add_argument("--doc-id", help="Document ID (default: stem of file name)")
    parser.add_argument(
        "--output-root",
        default="ingested",
        help="Root folder for saving images (default: 'ingested')",
    )
    args = parser.parse_args()

    pdf_path = args.pdf_path
    doc_id = args.doc_id or Path(pdf_path).stem

    print(f"Extracting images from {pdf_path}...")
    images = extract_images(
        file_path=pdf_path,
        doc_id=doc_id,
        output_root=args.output_root,
    )

    print(f"Extracted {len(images)} images.")
    data = [im.to_dict() for im in images]
    print(json.dumps(data, ensure_ascii=False, indent=2))