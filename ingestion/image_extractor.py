from __future__ import annotations

"""
image_extractor.py

หน้าที่:
- เปิดไฟล์ PDF
- ดึงรูปภาพทุกภาพในเอกสาร (ใช้ Docling)
- บันทึกลงโฟลเดอร์ (เช่น ingested/{doc_id}/images/img_001_001.png)
- [NEW] ใช้ Custom Vision Model (Qwen-VL) อ่านภาพและสร้าง Caption
- แปลงผลลัพธ์เป็น list[ImageBlock] ตาม schema
"""

import time
import base64
from pathlib import Path
from typing import List, Optional

# [CHANGE] เลิกใช้ fitz ในการดึงรูป (แต่ใช้ Docling แทน)
# import fitz  <-- ลบทิ้งได้เลย หรือ comment ไว้
from openai import OpenAI
from PIL import Image

from .schema import ImageBlock
# [NEW] เรียกใช้ Class Docling ที่เราเพิ่งสร้าง
from .docling_parser import DoclingImageParser

from dotenv import load_dotenv
import os

load_dotenv()

# [CHANGE] เลือกโมเดลที่เหมาะสมที่สุดสำหรับงาน Vision จากลิสต์
VISION_MODEL_NAME = "qwen/qwen2.5-vl-32b-instruct"

# -------------------------------------------------------------------
# Helper: Vision API (คงเดิม ไม่แตะต้อง)
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
    """ส่งรูปไปให้ AI อธิบาย (Captioning) - ฟังก์ชันเดิมที่ทำงานดีอยู่แล้ว"""
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
# Main Extraction (ส่วนที่แก้ไข Logic การดึงรูป)
# -------------------------------------------------------------------

def extract_images(
    file_path: str | Path,
    doc_id: str,
    output_root: str | Path = "ingested",
) -> List[ImageBlock]:
    """
    ดึงรูปภาพทั้งหมดจาก PDF (ใช้ Docling) และสร้าง Caption ด้วย AI
    """
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"PDF file not found: {path}")

    # กำหนดโฟลเดอร์ปลายทาง: ingested/{doc_id}/images
    output_root = Path(output_root)
    image_dir = output_root / doc_id / "images"
    image_dir.mkdir(parents=True, exist_ok=True)

    # Init AI Model (เหมือนเดิม)
    client, model_name = _get_vision_client()

    image_blocks: List[ImageBlock] = []

    # ----------------------------------------------------
    # [MODIFIED] ใช้ Docling แทน fitz
    # ----------------------------------------------------
    print(f"[image_extractor] Starting Docling extraction for {doc_id}...")
    parser = DoclingImageParser()
    
    # Docling จะจัดการเรื่องการวนลูปหน้าและเซฟไฟล์ให้เราเองในขั้นตอนนี้
    extracted_data = parser.extract_images(str(path), str(image_dir))
    
    # วนลูปข้อมูลรูปที่ได้จาก Docling
    for i, item in enumerate(extracted_data):
        file_path_on_disk = Path(item["file_path"])
        filename = item["filename"]
        page_number = item["page"]
        bbox = item["bbox"] # ได้ BBox แม่นยำกว่าเดิม
        
        # สร้าง ID
        img_id = f"img_{doc_id}_{i+1:04d}"
        
        # [KEEP] AI Captioning Logic เดิม (สำคัญมาก)
        caption_text = ""
        if client:
            print(f"[image_extractor] Generating caption for {filename} using {model_name}...")
            caption_text = _generate_image_caption(client, model_name, file_path_on_disk)
            
            # [KEEP] ใส่ delay กัน Rate Limit ตามที่คุณต้องการ (15s)
            print("   💤 Cooling down API for 15s...") 
            time.sleep(15)

        # สร้าง Object ImageBlock ตาม Schema เดิม
        image_block = ImageBlock(
            id=img_id,
            doc_id=doc_id,
            page=page_number,
            file_path=str(file_path_on_disk), # Path สำหรับให้ Backend/Frontend เรียกใช้
            caption=caption_text, 
            section=None,
            category="figure", # Docling ส่วนใหญ่ดึง Figure ออกมา
            bbox=bbox,
            extra={
                "source": "docling",
                "original_filename": filename,
                "ai_captioned": bool(caption_text)
            },
        )
        image_blocks.append(image_block)

    print(f"[image_extractor] Processed {len(image_blocks)} images for {doc_id}.")
    return image_blocks


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