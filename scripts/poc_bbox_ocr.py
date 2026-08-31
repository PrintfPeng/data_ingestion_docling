"""
scripts/poc_bbox_ocr.py

Proof-of-concept: compare two OCR approaches on ONE PDF:
- (A) Current: whole-page image → Gemini OCR → full text per page
- (B) New:     Docling detects text bboxes → annotated page → Gemini OCRs each labeled box

Prints side-by-side outputs + timing + char counts.

Usage inside container:
    docker exec ingestion-backend python scripts/poc_bbox_ocr.py <pdf_path> [page_limit]
Defaults: uses uploads/สัญญาจ้าง_นายอมรเทพ.pdf, first 3 pages
"""
from __future__ import annotations

import sys
import time
import logging
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ingestion.openrouter_ocr import OpenRouterVisionOCR
from ingestion.gemini_bbox_ocr import ocr_page_with_bboxes

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("poc")


# --------------- Docling helpers ---------------

def _docling_convert(pdf_path: str, do_ocr: bool = True):
    """Convert PDF via Docling. do_ocr=True lets Docling detect bboxes AND
    populate text — we can use the bboxes even if we plan to re-OCR the text.
    Set do_ocr=False for pure layout detection (faster but bboxes may be sparse
    for scanned PDFs)."""
    from docling.document_converter import DocumentConverter, PdfFormatOption
    from docling.datamodel.base_models import InputFormat
    from docling.datamodel.pipeline_options import PdfPipelineOptions

    opts = PdfPipelineOptions()
    opts.do_ocr = do_ocr
    opts.do_table_structure = False
    opts.generate_page_images = True
    opts.images_scale = 3.0
    conv = DocumentConverter(format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=opts)})
    return conv.convert(pdf_path).document


def _get_page_images(doc):
    imgs = {}
    for page_no, page in doc.pages.items():
        img = None
        if hasattr(page, "image") and page.image:
            if hasattr(page.image, "pil_image"):
                img = page.image.pil_image
            elif hasattr(page.image, "image"):
                img = page.image.image
        if img is None:
            try:
                img = page.get_image(scale=3.0)
            except Exception as e:
                logger.warning(f"failed to render page {page_no}: {e}")
        if img:
            imgs[page_no] = img
    return imgs


def _get_bboxes_per_page(doc):
    """Iterate all TextItems in Docling doc, group by page, return bbox list
    per page. BBoxes are converted from Docling's coord system to image pixel
    space (top-left origin, y-down).
    """
    per_page = {}
    counter = 0
    for item, _level in doc.iterate_items():
        if item.__class__.__name__ != "TextItem":
            continue
        if not getattr(item, "prov", None):
            continue
        page_no = item.prov[0].page_no
        raw_bbox = item.prov[0].bbox
        # Docling BoundingBox has: l, t, r, b + coord_origin
        # coord_origin can be "BOTTOMLEFT" (PDF standard) or "TOPLEFT"
        try:
            l, t, r, b = float(raw_bbox.l), float(raw_bbox.t), float(raw_bbox.r), float(raw_bbox.b)
        except AttributeError:
            try:
                l, t, r, b = raw_bbox.as_tuple()
            except Exception:
                continue
        coord_origin = getattr(raw_bbox, "coord_origin", None)
        # Docling's page size (points)
        page = doc.pages.get(page_no)
        pdf_h = float(getattr(page.size, "height", 0)) if page else 0.0

        counter += 1
        per_page.setdefault(page_no, []).append({
            "id": f"b_{counter}",
            "_l": l, "_t": t, "_r": r, "_b": b,
            "_coord_origin": str(coord_origin),
            "_pdf_h": pdf_h,
            "text_dl": (item.text or "").strip()[:120],  # Docling's OCR text for reference
        })
    return per_page


def _pdf_bbox_to_image(bbox_dict, img_w: int, img_h: int, pdf_w: float, pdf_h: float):
    """Convert PDF-space bbox → image pixel coords (top-left origin)."""
    sx = img_w / pdf_w if pdf_w > 0 else 1.0
    sy = img_h / pdf_h if pdf_h > 0 else 1.0
    l, t, r, b = bbox_dict["_l"], bbox_dict["_t"], bbox_dict["_r"], bbox_dict["_b"]

    origin = (bbox_dict.get("_coord_origin") or "").upper()
    if "BOTTOM" in origin:
        # BOTTOMLEFT: flip y
        y0 = (pdf_h - t) * sy
        y1 = (pdf_h - b) * sy
    else:
        # TOPLEFT already
        y0, y1 = t * sy, b * sy

    x0, x1 = l * sx, r * sx
    if y0 > y1:
        y0, y1 = y1, y0
    if x0 > x1:
        x0, x1 = x1, x0
    return (x0, y0, x1, y1)


# --------------- Approaches ---------------

def approach_a(pdf_path: str, page_limit: int) -> list:
    print("\n===== Approach A: Whole-page OCR (current pipeline) =====")
    t0 = time.time()
    ocr = OpenRouterVisionOCR()
    pages = ocr.ocr_pdf(pdf_path)
    elapsed = time.time() - t0
    pages = pages[:page_limit]
    print(f"[A] {len(pages)} pages · {elapsed:.1f}s wall")
    return [{"page": p["page"], "text": p["content"]} for p in pages]


def approach_b(pdf_path: str, page_limit: int) -> list:
    print("\n===== Approach B: BBox-guided (Docling layout + Gemini OCR per box) =====")
    t0 = time.time()
    doc = _docling_convert(pdf_path, do_ocr=True)  # need bboxes; Docling's own text is ignored
    print(f"[B] docling parse: {time.time()-t0:.1f}s")

    imgs = _get_page_images(doc)
    per_page = _get_bboxes_per_page(doc)

    results = []
    for page_no in sorted(per_page.keys())[:page_limit]:
        img = imgs.get(page_no)
        if img is None:
            continue
        page = doc.pages[page_no]
        pdf_w, pdf_h = float(page.size.width), float(page.size.height)
        img_w, img_h = img.size

        raw_bboxes = per_page[page_no]
        bboxes_img = []
        for b in raw_bboxes:
            bb = _pdf_bbox_to_image(b, img_w, img_h, pdf_w, pdf_h)
            bboxes_img.append({"id": b["id"], "bbox": bb, "docling_text": b["text_dl"]})

        print(f"[B] page {page_no}: {len(bboxes_img)} bboxes · sending to Gemini...")
        t_page = time.time()
        items = ocr_page_with_bboxes(img, bboxes_img)
        print(f"    ← received {len(items)} items · {time.time()-t_page:.1f}s")

        combined = "\n".join(it["text"] for it in items if it.get("text"))
        results.append({
            "page": page_no,
            "text": combined,
            "items": items,
            "bboxes": bboxes_img,
        })

    print(f"[B] total {time.time()-t0:.1f}s")
    return results


def main():
    pdf = sys.argv[1] if len(sys.argv) > 1 else "/app/uploads/สัญญาจ้าง_นายอมรเทพ.pdf"
    page_limit = int(sys.argv[2]) if len(sys.argv) > 2 else 3

    if not Path(pdf).exists():
        print(f"PDF not found: {pdf}")
        sys.exit(1)
    print(f"PDF: {pdf}")
    print(f"Limit: first {page_limit} pages\n")

    pages_a = approach_a(pdf, page_limit)
    pages_b = approach_b(pdf, page_limit)

    # Side-by-side comparison
    print("\n\n" + "=" * 80)
    print("SIDE-BY-SIDE COMPARISON")
    print("=" * 80)
    by_page_b = {p["page"]: p for p in pages_b}
    for pa in pages_a:
        pb = by_page_b.get(pa["page"])
        print(f"\n----- PAGE {pa['page']} -----")
        print(f"  Approach A: {len(pa['text'])} chars")
        print(f"  Approach B: {len(pb['text']) if pb else 0} chars ({len(pb['items']) if pb else 0} bboxes)")
        print(f"\n  [A] FIRST 400 CHARS:")
        print("  " + pa["text"][:400].replace("\n", "\n  "))
        if pb:
            print(f"\n  [B] FIRST 400 CHARS:")
            print("  " + pb["text"][:400].replace("\n", "\n  "))
            print(f"\n  [B] BBox breakdown:")
            for it in pb["items"][:8]:
                print(f"    · {it['id']}: {it['text'][:100]}")


if __name__ == "__main__":
    main()
