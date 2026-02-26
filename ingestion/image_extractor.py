from __future__ import annotations

"""
image_extractor.py (Hybrid Edition)

หน้าที่:
- เปิดไฟล์ PDF
- ดึงรูปภาพทุกภาพในเอกสาร (ใช้ Docling)
- [NEW] ระบบ AI 3 ชั้น (OpenRouter -> Google -> Local)
- แปลงผลลัพธ์เป็น list[ImageBlock] ตาม schema
"""

import time
import base64
import os
from pathlib import Path
from typing import List, Optional

from openai import OpenAI
from PIL import Image
from dotenv import load_dotenv

from .schema import ImageBlock
# [KEEP] ใช้ Docling เหมือนเดิม
from .docling_parser import DoclingImageParser

# [NEW] Import Google GenAI (สำหรับแผนสำรอง)
try:
    import google.generativeai as genai
    HAS_GOOGLE = True
except ImportError:
    HAS_GOOGLE = False

load_dotenv()

# Config
VISION_MODEL_NAME = "qwen/qwen2.5-vl-32b-instruct"

# -------------------------------------------------------------------
# Helper: AI Clients & Encoding
# -------------------------------------------------------------------

def _get_openai_client() -> tuple[Optional[OpenAI], Optional[str]]:
    """เตรียม OpenAI Client (OpenRouter)"""
    api_key = os.getenv("CUSTOM_API_KEY")
    base_url = os.getenv("CUSTOM_API_BASE")
    
    if not api_key:
        # [DEBUG] แจ้งเตือนถ้าไม่มี Key
        # print("   ⚠️ [Config] CUSTOM_API_KEY not found. Plan A disabled.") 
        return None, None
        
    try:
        client = OpenAI(api_key=api_key, base_url=base_url)
        return client, VISION_MODEL_NAME
    except: return None, None

def _get_google_client():
    """เตรียม Google Gemini Client"""
    if not HAS_GOOGLE: return None
    api_key = os.getenv("GOOGLE_API_KEY")
    
    if not api_key:
        # [DEBUG] แจ้งเตือนถ้าไม่มี Key
        # print("   ⚠️ [Config] GOOGLE_API_KEY not found. Plan B disabled.")
        return None
        
    try:
        genai.configure(api_key=api_key)
        # ใช้โมเดล Flash เพราะฟรีและเร็ว
        return genai.GenerativeModel('gemini-2.0-flash')
    except: return None

def _encode_image(image_path: Path) -> str:
    """แปลงไฟล์รูปเป็น Base64"""
    try:
        # [FIX] ใช้ Absolute Path เสมอ เพื่อป้องกัน Error หาไฟล์ไม่เจอ
        abs_path = image_path.resolve()
        with open(abs_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')
    except Exception as e:
        print(f"   ❌ Error encoding image {image_path}: {e}")
        return ""

# -------------------------------------------------------------------
# Logic: Generate Caption (Hybrid Fallback System)
# -------------------------------------------------------------------

def _generate_caption_hybrid(image_path: Path) -> str:
    """
    ระบบสร้างคำบรรยายภาพแบบ 3 ชั้น:
    1. OpenRouter (Qwen) -> ถ้าได้ พัก 15 วิ
    2. Google Gemini     -> ถ้าได้ พัก 2 วิ
    3. Empty String      -> (Local Fallback)
    """
    
    # [FIX] ตรวจสอบไฟล์ก่อนส่ง AI
    if not image_path.exists():
        print(f"   ❌ Image file missing: {image_path}")
        return ""

    prompt = (
        "อธิบายรูปภาพนี้โดยละเอียด: "
        "1. ถ้าเป็นกราฟ/แผนภูมิ ให้บอกชื่อแกน ตัวเลขสำคัญ และแนวโน้ม "
        "2. ถ้าเป็นรูปถ่าย/ไดอะแกรม ให้บอกว่าคืออะไรและมีองค์ประกอบสำคัญอะไรบ้าง "
        "3. ถ้ามีข้อความในภาพ ให้อ่านและสรุปข้อความนั้นมาด้วย "
        "ตอบเป็นภาษาไทย กระชับและได้ใจความ"
    )

    # --- PLAN A: OpenRouter (Qwen) ---
    openai_client, model_name = _get_openai_client()
    if openai_client:
        try:
            # print(f"   [AI] Trying Plan A: {model_name}...")
            base64_image = _encode_image(image_path)
            if base64_image:
                response = openai_client.chat.completions.create(
                    model=model_name,
                    messages=[
                        {
                            "role": "user",
                            "content": [
                                {"type": "text", "text": prompt},
                                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{base64_image}"}},
                            ],
                        }
                    ],
                    max_tokens=300,
                    timeout=60
                )
                caption = response.choices[0].message.content.strip()
                
                print("   💤 Cooling down Plan A (15s)...")
                time.sleep(15)
                return caption
            
        except Exception as e:
            print(f"   ⚠️ Plan A (OpenRouter) failed: {e}")

    # --- PLAN B: Google Gemini ---
    google_model = _get_google_client()
    if google_model:
        try:
            print(f"   [AI] 🔄 Switching to Plan B: Google Gemini...")
            img = Image.open(image_path)
            response = google_model.generate_content([prompt, img])
            caption = response.text.strip()
            
            # [LOGIC] Google เร็วและฟรี พักนิดเดียวพอ (2s)
            time.sleep(2)
            return caption
            
        except Exception as e:
            print(f"   ⚠️ Plan B (Google) failed: {e}")

    # --- PLAN C: Local (ยอมแพ้) ---
    print("   [AI] ❌ All AI failed. Skipping caption.")
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
    ดึงรูปภาพ (Docling) + สร้าง Caption (Hybrid AI)
    """
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"PDF file not found: {path}")

    # กำหนดโฟลเดอร์ปลายทาง
    output_root = Path(output_root)
    image_dir = output_root / doc_id / "images"
    image_dir.mkdir(parents=True, exist_ok=True)

    # 1. ใช้ Docling ดึงรูป (ส่วนเดิม ไม่แก้)
    print(f"[image_extractor] Starting Docling extraction for {doc_id}...")
    parser = DoclingImageParser()
    extracted_data = parser.extract_images(str(path), str(image_dir))
    
    image_blocks: List[ImageBlock] = []

    # 2. วนลูปรูปที่ได้
    for i, item in enumerate(extracted_data):
        # [FIX] ใช้ resolve() เพื่อให้มั่นใจว่าเป็น Absolute Path ก่อนส่งให้ AI
        file_path_on_disk = Path(item["file_path"]).resolve()
        file_name = item["file_name"] 
        
        img_id = f"img_{doc_id}_{item['index']+1:04d}"
        
        print(f"[image_extractor] Generating caption for {file_name}...")
        
        # ส่ง Path เต็มไปให้ฟังก์ชันสร้าง Caption
        caption_text = _generate_caption_hybrid(file_path_on_disk)
        
        # สร้าง Object ImageBlock
        image_block = ImageBlock(
            id=img_id,
            doc_id=doc_id,
            page=item["page"],
            # เก็บ Path เป็น Relative เหมือนเดิมเพื่อให้ Frontend ใช้ง่าย
            file_path=item["file_path"], 
            caption=caption_text, 
            section=None,
            category="figure",
            bbox=item["bbox"],
            extra={
                "source": "docling",
                "original_file_name": file_name,
                "ai_captioned": bool(caption_text),
                "ai_model": "hybrid"
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