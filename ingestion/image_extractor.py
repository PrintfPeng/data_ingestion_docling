# ingestion/image_extractor.py

from __future__ import annotations

import time
from pathlib import Path
from typing import List, Optional, Union

import fitz  # PyMuPDF
from google import genai 
from PIL import Image

try:
    from .schema import ImageBlock
    from .config import GOOGLE_API_KEY
except ImportError:
    from ingestion.schema import ImageBlock
    from ingestion.config import GOOGLE_API_KEY

def _get_gemini_vision_client():
    if not GOOGLE_API_KEY:
        print("[image_extractor] Warning: No GOOGLE_API_KEY.")
        return None
    try:
        return genai.Client(api_key=GOOGLE_API_KEY)
    except Exception as e:
        print(f"[image_extractor] Failed to init Gemini: {e}")
        return None

def generate_image_description_md(client, image_input: Union[str, Path, Image.Image]) -> str:
    if not client: return ""

    try:
        if isinstance(image_input, (str, Path)):
            img = Image.open(image_input)
        else:
            img = image_input

        prompt = (
            "Analyze this image from a document. Provide a detailed description for retrieval purposes.\n"
            "Output Format in Markdown:\n"
            "1. **Type**: (e.g., Bar Chart, Flowchart, Photograph, Table)\n"
            "2. **Text Content**: Transcribe visible text exactly.\n"
            "3. **Description**: Describe relationships, trends, or steps.\n"
            "4. **Keywords**: List 5-10 specific keywords (Thai/English).\n"
        )

        # [UPDATED] เปลี่ยนชื่อโมเดลเป็นรุ่นที่เสถียรกว่า (Gemini 2.0 Flash)
        response = client.models.generate_content(
            model="gemini-2.0-flash", 
            contents=[prompt, img]
        )
        
        return response.text.strip() if response.text else ""
    except Exception as e:
        print(f"[image_extractor] Caption failed: {e}")
        return ""

# ... (ส่วน extract_images ด้านล่างยังคงเดิมเหมือนไฟล์ก่อนหน้า) ...
# เพื่อความครบถ้วน ผมใส่ส่วน extract_images ให้ด้วยด้านล่างนี้ครับ

def extract_images(
    file_path: str | Path,
    doc_id: str,
    output_root: str | Path = "ingested",
) -> List[ImageBlock]:
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"PDF file not found: {path}")

    output_root = Path(output_root)
    image_dir = output_root / doc_id / "images"
    image_dir.mkdir(parents=True, exist_ok=True)

    client = _get_gemini_vision_client()

    pdf_doc = fitz.open(path)
    image_blocks: List[ImageBlock] = []

    try:
        image_counter = 0

        for page_index in range(pdf_doc.page_count):
            page = pdf_doc[page_index]
            page_number = page_index + 1
            images = page.get_images(full=True)

            for img_index, img in enumerate(images, start=1):
                xref = img[0]
                try:
                    base_image = pdf_doc.extract_image(xref)
                except Exception as e:
                    continue

                img_bytes: bytes = base_image["image"]
                img_ext: str = base_image.get("ext", "png")
                width: int = base_image.get("width", 0)
                height: int = base_image.get("height", 0)

                if width < 50 or height < 50:
                    continue

                image_counter += 1
                img_id = f"img_{image_counter:04d}"
                filename = f"img_p{page_number:03d}_{img_index:03d}.{img_ext}"
                file_path_on_disk = image_dir / filename

                with open(file_path_on_disk, "wb") as f:
                    f.write(img_bytes)
                
                caption_text = ""
                if client:
                    print(f"[image_extractor] Generating MD caption for {filename}...")
                    caption_text = generate_image_description_md(client, file_path_on_disk)
                    time.sleep(1.0) 

                image_block = ImageBlock(
                    id=img_id,
                    doc_id=doc_id,
                    page=page_number,
                    file_path=str(file_path_on_disk),
                    caption=caption_text, 
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

    parser = argparse.ArgumentParser()
    parser.add_argument("pdf_path")
    parser.add_argument("--doc-id")
    parser.add_argument("--output-root", default="ingested")
    args = parser.parse_args()

    pdf_path = args.pdf_path
    doc_id = args.doc_id or Path(pdf_path).stem

    print(f"Extracting images from {pdf_path}...")
    images = extract_images(
        file_path=pdf_path,
        doc_id=doc_id,
        output_root=args.output_root,
    )
    data = [im.to_dict() for im in images]
    print(json.dumps(data, ensure_ascii=False, indent=2))