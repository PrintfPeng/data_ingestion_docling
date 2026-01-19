from __future__ import annotations

"""
image_extractor.py

หน้าที่:
- เปิดไฟล์ PDF
- ดึงรูปภาพทุกภาพในเอกสาร
- บันทึกลงโฟลเดอร์ (เช่น ingested/{doc_id}/images/img_001_001.png)
- [NEW] ใช้ Custom Vision Model (Qwen-VL) อ่านภาพและสร้าง Caption
- แปลงผลลัพธ์เป็น list[ImageBlock] ตาม schema
"""

import time
import base64
from pathlib import Path
from typing import List, Optional

import fitz  # PyMuPDF
# [CHANGE] ใช้ OpenAI Client สำหรับ Custom API
from openai import OpenAI
from PIL import Image

from .schema import ImageBlock

from dotenv import load_dotenv
import os

load_dotenv()

# [CHANGE] เลือกโมเดลที่เหมาะสมที่สุดสำหรับงาน Vision จากลิสต์
# qwen2.5-vl-32b-instruct เป็นโมเดล Vision-Language ที่เก่งมาก
VISION_MODEL_NAME = "qwen/qwen2.5-vl-32b-instruct"

# -------------------------------------------------------------------
# Helper: Vision API
# -------------------------------------------------------------------

def _get_vision_client() -> tuple[Optional[OpenAI], Optional[str]]:
    """เตรียม OpenAI Client สำหรับงาน Vision"""
    api_key = os.getenv("CUSTOM_API_KEY")
    base_url = os.getenv("CUSTOM_API_BASE")

    if not api_key:
        print("[image_extractor] Warning: No CUSTOM_API_KEY. Image captioning will be skipped.")
        return None, None

    try:
        client = OpenAI(
            api_key=api_key,
            base_url=base_url
        )
        return client, VISION_MODEL_NAME
    except Exception as e:
        print(f"[image_extractor] Failed to init OpenAI Client: {e}")
        return None, None

def _encode_image(image_path: Path) -> str:
    """แปลงไฟล์รูปเป็น Base64"""
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

def _generate_image_caption(client: OpenAI, model_name: str, image_path: Path) -> str:
    """ส่งรูปไปให้ AI อธิบาย (Captioning)"""
    if not client:
        return ""

    try:
        # Encode รูปเป็น Base64
        base64_image = _encode_image(image_path)
        
        prompt = (
            "อธิบายรูปภาพนี้โดยละเอียด: "
            "1. ถ้าเป็นกราฟ/แผนภูมิ ให้บอกชื่อแกน ตัวเลขสำคัญ และแนวโน้ม "
            "2. ถ้าเป็นรูปถ่าย/ไดอะแกรม ให้บอกว่าคืออะไรและมีองค์ประกอบสำคัญอะไรบ้าง "
            "3. ถ้ามีข้อความในภาพ ให้อ่านและสรุปข้อความนั้นมาด้วย "
            "ตอบเป็นภาษาไทย กระชับและได้ใจความ"
        )

        response = client.chat.completions.create(
            model=model_name,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/png;base64,{base64_image}"
                            },
                        },
                    ],
                }
            ],
            max_tokens=300,
        )
        
        return response.choices[0].message.content.strip()
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
    client, model_name = _get_vision_client()

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
                if client:
                    print(f"[image_extractor] Generating caption for {filename} using {model_name}...")
                    caption_text = _generate_image_caption(client, model_name, file_path_on_disk)
                    # ใส่ delay นิดหน่อยกัน Rate Limit (ถ้าจำเป็น)
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