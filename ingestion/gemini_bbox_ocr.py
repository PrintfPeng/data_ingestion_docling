"""
ingestion/gemini_bbox_ocr.py

BBox-guided OCR (Approach 2 POC).

Given a page image + a list of bounding boxes (from Docling layout detection),
annotate the image with numbered red rectangles, send to Gemini, and ask it to
OCR text inside each labeled box. Returns text per bbox in original order.

Advantages over whole-page OCR:
- Reading order preserved from Docling's layout detection
- Gemini sees the whole page (rich context) but focuses on labeled regions
- Bbox structure helps disambiguate similar-looking dates/numbers on same page
- Single API call per page (unlike per-region cropping)
"""
from __future__ import annotations

import os
import io
import re
import json
import base64
import logging
from typing import List, Dict, Optional, Tuple, Any

from PIL import Image, ImageDraw, ImageFont
from openai import OpenAI

logger = logging.getLogger(__name__)

VISION_API_KEY = os.getenv("VISION_API_KEY")
VISION_API_BASE = os.getenv("VISION_API_BASE", "https://openrouter.ai/api/v1")
VISION_MODEL = os.getenv("VISION_MODEL", "google/gemini-2.5-flash")

# Rendering
BOX_COLOR = "red"
BOX_WIDTH = 3
LABEL_BG = (255, 255, 255)
LABEL_FG = (200, 0, 0)


BBOX_PROMPT = """ภาพนี้คือหน้า PDF ที่มีกล่อง (bounding boxes) สีแดงพร้อมป้ายกำกับ (id) เช่น b_1, b_2, b_3, ...

หน้าที่ของคุณ:
1. OCR ข้อความในแต่ละกล่อง — โดยอ่านเฉพาะข้อความที่อยู่ **ภายในกล่อง** เท่านั้น
2. รักษาเลขไทย/สระ/วรรณยุกต์/สัญลักษณ์พิเศษตามที่เห็นในภาพ
3. ถ้ากล่องว่างเปล่า/อ่านไม่ได้ ใส่ text เป็น ""
4. ห้าม OCR ข้อความนอกกล่อง ห้ามเพิ่ม id ที่ไม่ได้มีในภาพ ห้ามข้าม id ที่มี
5. ส่งกลับเป็น JSON array เท่านั้น (ห้ามมีคำอธิบายอื่น) ตาม schema นี้:
[{"id": "b_1", "text": "..."}, {"id": "b_2", "text": "..."}, ...]

รักษาลำดับ id เหมือนที่เห็นในภาพ (bottom-up หรือ top-down ตามลำดับตัวเลข)"""


def _image_to_b64(img: Image.Image, format: str = "PNG") -> str:
    buf = io.BytesIO()
    img.save(buf, format=format)
    return base64.b64encode(buf.getvalue()).decode("ascii")


def _load_font(size: int = 20):
    for path in [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
    ]:
        try:
            return ImageFont.truetype(path, size=size)
        except Exception:
            continue
    return ImageFont.load_default()


def annotate_page(img: Image.Image, bboxes: List[Dict[str, Any]]) -> Image.Image:
    """Draw numbered red rectangles on a copy of the page image.

    bboxes: [{"id": "b_1", "bbox": (x0, y0, x1, y1)}, ...]  — image pixel coords
    """
    annotated = img.copy().convert("RGB")
    draw = ImageDraw.Draw(annotated)
    font = _load_font(size=22)

    for b in bboxes:
        x0, y0, x1, y1 = [int(round(v)) for v in b["bbox"]]
        draw.rectangle([x0, y0, x1, y1], outline=BOX_COLOR, width=BOX_WIDTH)

        label = b.get("id", "?")
        # Small white background for the label so it's readable
        try:
            tw, th = draw.textbbox((0, 0), label, font=font)[2:]
        except Exception:
            tw, th = 40, 20
        pad = 3
        # Place label above the box, or inside if it would go off-image
        ly0 = max(0, y0 - th - 2 * pad)
        ly1 = ly0 + th + 2 * pad
        lx0 = x0
        lx1 = min(annotated.width, x0 + tw + 2 * pad)
        draw.rectangle([lx0, ly0, lx1, ly1], fill=LABEL_BG, outline=BOX_COLOR)
        draw.text((lx0 + pad, ly0 + pad), label, fill=LABEL_FG, font=font)

    return annotated


def ocr_page_with_bboxes(
    page_image: Image.Image,
    bboxes: List[Dict[str, Any]],
    model: Optional[str] = None,
    max_tokens: int = 4096,
) -> List[Dict[str, str]]:
    """Annotate + send to Gemini. Returns [{"id":..., "text":...}, ...].

    Returns empty list on failure (caller can fall back to whole-page OCR).
    """
    if not bboxes:
        return []
    if not VISION_API_KEY:
        raise RuntimeError("VISION_API_KEY not set")

    annotated = annotate_page(page_image, bboxes)
    b64 = _image_to_b64(annotated)

    client = OpenAI(api_key=VISION_API_KEY, base_url=VISION_API_BASE)
    try:
        resp = client.chat.completions.create(
            model=model or VISION_MODEL,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": BBOX_PROMPT},
                        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}},
                    ],
                }
            ],
            temperature=0.1,
            max_tokens=max_tokens,
        )
        text = (resp.choices[0].message.content or "").strip()
    except Exception as e:
        logger.error(f"[bbox_ocr] API call failed: {e}")
        return []

    # Extract JSON array — tolerant to markdown fences / prose wrapping
    m = re.search(r"\[[\s\S]*\]", text)
    if not m:
        logger.warning(f"[bbox_ocr] no JSON array in response · first 200 chars: {text[:200]}")
        return []
    try:
        arr = json.loads(m.group(0))
    except json.JSONDecodeError as e:
        logger.warning(f"[bbox_ocr] JSON parse failed: {e} · text: {text[:300]}")
        return []

    out = []
    for item in arr:
        if isinstance(item, dict) and "id" in item:
            out.append({"id": str(item["id"]), "text": (item.get("text") or "").strip()})
    return out
