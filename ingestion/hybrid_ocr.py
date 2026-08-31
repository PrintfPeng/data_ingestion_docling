"""
ingestion/hybrid_ocr.py

Hybrid OCR router: combines the two POC approaches so each page gets the best
OCR strategy for its layout.

- **Approach A** (whole-page image → Gemini): simple, robust, works even when
  layout detection misses regions. Best for single-column body text.
- **Approach B** (Docling bboxes + annotated page → Gemini per-box JSON):
  preserves structural granularity (each date, heading, table cell as its own
  labeled item). Best for infographics / multi-column / timelines.

Routing rule per page:
  bboxes ≥ HYBRID_BBOX_MIN  →  try B; if B returns empty → fall back to A
  bboxes  < HYBRID_BBOX_MIN →  A directly

This lets us avoid the failure mode where B silently drops entire pages
because Docling's layout detector produces 0 bboxes.
"""
from __future__ import annotations

import os
import base64
import logging
from io import BytesIO
from typing import List, Dict, Any, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed

from PIL import Image

logger = logging.getLogger(__name__)

# --- Config ---
HYBRID_BBOX_MIN = int(os.getenv("HYBRID_BBOX_MIN", "3"))
HYBRID_WORKERS = int(os.getenv("HYBRID_WORKERS", "5"))

# Lazy import to keep module loadable without vision deps
_OCR_A_INSTANCE = None
_OCR_B_FN = None


def _get_a():
    global _OCR_A_INSTANCE
    if _OCR_A_INSTANCE is None:
        from .openrouter_ocr import OpenRouterVisionOCR
        _OCR_A_INSTANCE = OpenRouterVisionOCR()
    return _OCR_A_INSTANCE


def _get_b():
    global _OCR_B_FN
    if _OCR_B_FN is None:
        from .gemini_bbox_ocr import ocr_page_with_bboxes
        _OCR_B_FN = ocr_page_with_bboxes
    return _OCR_B_FN


# --- BBox extraction from Docling doc ---
def _get_bboxes_per_page(doc) -> Dict[int, List[Dict[str, Any]]]:
    """Iterate all TextItems in Docling doc, group by page_no.
    Returns {page_no: [{id, bbox_pdf, coord_origin, pdf_h}, ...]}"""
    per_page: Dict[int, List[Dict[str, Any]]] = {}
    counter = 0
    for item, _level in doc.iterate_items():
        if item.__class__.__name__ != "TextItem":
            continue
        prov = getattr(item, "prov", None)
        if not prov:
            continue
        page_no = prov[0].page_no
        raw = prov[0].bbox
        try:
            l = float(getattr(raw, "l", 0))
            t = float(getattr(raw, "t", 0))
            r = float(getattr(raw, "r", 0))
            b = float(getattr(raw, "b", 0))
        except Exception:
            continue
        page = doc.pages.get(page_no)
        pdf_h = float(getattr(page.size, "height", 0)) if page else 0.0
        counter += 1
        per_page.setdefault(page_no, []).append({
            "id": f"b_{counter}",
            "_l": l, "_t": t, "_r": r, "_b": b,
            "_coord_origin": str(getattr(raw, "coord_origin", "")),
            "_pdf_h": pdf_h,
        })
    return per_page


def _pdf_bbox_to_image(bbox_dict, img_w: int, img_h: int, pdf_w: float, pdf_h: float):
    """Convert PDF-space bbox to image-pixel coords (top-left origin, y-down)."""
    sx = img_w / pdf_w if pdf_w > 0 else 1.0
    sy = img_h / pdf_h if pdf_h > 0 else 1.0
    l, t, r, b = bbox_dict["_l"], bbox_dict["_t"], bbox_dict["_r"], bbox_dict["_b"]
    origin = (bbox_dict.get("_coord_origin") or "").upper()
    if "BOTTOM" in origin:
        y0 = (pdf_h - t) * sy
        y1 = (pdf_h - b) * sy
    else:
        y0, y1 = t * sy, b * sy
    x0, x1 = l * sx, r * sx
    if y0 > y1: y0, y1 = y1, y0
    if x0 > x1: x0, x1 = x1, x0
    return (x0, y0, x1, y1)


# --- Per-page router ---
def _ocr_one_page(
    page_no: int,
    page_image: Image.Image,
    raw_bboxes: List[Dict[str, Any]],
    pdf_w: float,
    pdf_h: float,
) -> Dict[str, Any]:
    """Route to A or B based on bbox count; fall back to A if B fails."""
    img_w, img_h = page_image.size
    n_bboxes = len(raw_bboxes)

    # Route: enough bboxes → try B
    if n_bboxes >= HYBRID_BBOX_MIN:
        bboxes_img = [
            {"id": b["id"], "bbox": _pdf_bbox_to_image(b, img_w, img_h, pdf_w, pdf_h)}
            for b in raw_bboxes
        ]
        try:
            items = _get_b()(page_image, bboxes_img)
        except Exception as e:
            logger.warning(f"[hybrid_ocr] page {page_no} B failed: {e}")
            items = []
        if items:
            text = "\n".join(it["text"] for it in items if it.get("text"))
            if text.strip():
                logger.info(f"[hybrid_ocr] page {page_no}: B ok · {n_bboxes} bboxes · {len(text)} chars")
                return {"page": page_no, "content": text, "strategy": "B"}
        logger.info(f"[hybrid_ocr] page {page_no}: B empty · fallback → A")

    # Fallback / default: A
    try:
        buf = BytesIO(); page_image.save(buf, format="PNG")
        b64 = base64.b64encode(buf.getvalue()).decode("ascii")
        text = _get_a()._ocr_page_with_retry(b64, page_no)
        logger.info(f"[hybrid_ocr] page {page_no}: A · {n_bboxes} bboxes · {len(text)} chars")
        return {"page": page_no, "content": text or "", "strategy": "A"}
    except Exception as e:
        logger.error(f"[hybrid_ocr] page {page_no}: A also failed: {e}")
        return {"page": page_no, "content": "", "strategy": "fail"}


# --- Public API ---
def hybrid_ocr_from_doc(doc, page_images: Dict[int, Image.Image]) -> List[Dict[str, Any]]:
    """OCR all pages using hybrid A+B routing.
    Args:
        doc: Docling document (already parsed)
        page_images: {page_no: PIL.Image} — Docling's rendered pages
    Returns:
        [{"page": int, "content": str, "strategy": "A"|"B"|"fail"}, ...]  in page order
    """
    if not page_images:
        return []

    per_page_bboxes = _get_bboxes_per_page(doc)
    page_sizes = {}
    for pn, p in doc.pages.items():
        try:
            page_sizes[pn] = (float(p.size.width), float(p.size.height))
        except Exception:
            page_sizes[pn] = (0.0, 0.0)

    # Run pages in parallel (OCR calls are I/O-bound; safe to parallelize)
    results: List[Optional[Dict[str, Any]]] = []
    order = sorted(page_images.keys())
    results = [None] * len(order)

    with ThreadPoolExecutor(max_workers=HYBRID_WORKERS) as pool:
        fut_to_idx = {}
        for i, page_no in enumerate(order):
            img = page_images[page_no]
            bboxes = per_page_bboxes.get(page_no, [])
            pdf_w, pdf_h = page_sizes.get(page_no, (0.0, 0.0))
            fut = pool.submit(_ocr_one_page, page_no, img, bboxes, pdf_w, pdf_h)
            fut_to_idx[fut] = i
        for fut in as_completed(fut_to_idx):
            i = fut_to_idx[fut]
            try:
                results[i] = fut.result()
            except Exception as e:
                logger.error(f"[hybrid_ocr] worker crash: {e}")
                results[i] = {"page": order[i], "content": "", "strategy": "fail"}

    # Summary
    strat_count = {}
    for r in results:
        s = r.get("strategy", "?") if r else "?"
        strat_count[s] = strat_count.get(s, 0) + 1
    logger.info(f"[hybrid_ocr] total {len(results)} pages · strategies: {strat_count}")

    return [r for r in results if r is not None]
