"""
ingestion/openrouter_ocr.py

OCR ผ่าน OpenRouter Vision LLM (Gemini 2.5 Flash by default).
- Parallel processing ด้วย ThreadPoolExecutor
- Retry with exponential backoff สำหรับ 429/network errors
- ใช้แทน text-layer ของ PDF ที่มี font mapping พังหรือเป็น scanned document
"""
import os
import time
import base64
import logging
import concurrent.futures
from pathlib import Path
from typing import List, Dict, Optional, Tuple

import fitz  # pymupdf
from openai import OpenAI

logger = logging.getLogger(__name__)


DEFAULT_MODEL = "google/gemini-2.5-flash"
DEFAULT_API_BASE = "https://openrouter.ai/api/v1"
DEFAULT_DPI = 300
DEFAULT_MAX_WORKERS = int(os.getenv("OCR_MAX_WORKERS", "5"))
DEFAULT_MAX_RETRIES = int(os.getenv("OCR_MAX_RETRIES", "3"))

OCR_PROMPT_TH = """คุณคือระบบ OCR สำหรับเอกสารภาษาไทย/อังกฤษ
โปรดถอดข้อความจากภาพนี้ทั้งหมดตามที่เห็น โดย:
1. คงลำดับ ย่อหน้า และการเว้นบรรทัดตามภาพจริง
2. ถ้ามีตาราง ให้แสดงเป็น markdown table
3. ถ้าเจอตราประทับ/ลายเซ็น/ตราครุฑ ให้ใส่ [รูปภาพ: <คำอธิบายสั้นๆ>]
4. ถ้าอ่านบางส่วนไม่ออก ให้ใส่ [อ่านไม่ออก]
5. ห้ามเพิ่มคำนำ/คำอธิบาย/ความคิดเห็นของตัวเอง — ส่งกลับเฉพาะข้อความที่ถอดได้เท่านั้น
"""


class OpenRouterVisionOCR:
    """OCR PDF pages ผ่าน OpenRouter vision model — พร้อม parallel + retry."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        api_base: Optional[str] = None,
        model: Optional[str] = None,
        dpi: int = DEFAULT_DPI,
        max_workers: int = DEFAULT_MAX_WORKERS,
        max_retries: int = DEFAULT_MAX_RETRIES,
    ):
        # Phase 5.4: subprocess-set override (per-user key) wins over the shared env
        self.api_key = api_key or os.getenv("VISION_API_KEY_OVERRIDE") or os.getenv("VISION_API_KEY")
        self.api_base = api_base or os.getenv("VISION_API_BASE", DEFAULT_API_BASE)
        self.model = model or os.getenv("VISION_MODEL", DEFAULT_MODEL)
        self.dpi = dpi
        self.max_workers = max_workers
        self.max_retries = max_retries

        if not self.api_key:
            raise RuntimeError("VISION_API_KEY not set. Cannot run OpenRouter OCR.")

        self.client = OpenAI(api_key=self.api_key, base_url=self.api_base)

    def is_available(self) -> bool:
        return bool(self.api_key)

    def _render_page(self, page: "fitz.Page") -> bytes:
        """Render a single PDF page as PNG bytes."""
        zoom = self.dpi / 72.0
        matrix = fitz.Matrix(zoom, zoom)
        pix = page.get_pixmap(matrix=matrix, alpha=False)
        return pix.tobytes("png")

    def _call_vision_api(self, image_b64: str, page_no: int = 0) -> str:
        """One call to the vision LLM. Raises on failure so caller can retry."""
        resp = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": OCR_PROMPT_TH},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/png;base64,{image_b64}"
                            },
                        },
                    ],
                }
            ],
            temperature=0.1,
            max_tokens=4096,
        )
        # Cost telemetry — best-effort, never blocks OCR
        try:
            from backend.services.cost_tracker import log_from_response
            log_from_response(
                endpoint="ocr",
                provider="api",
                model=self.model,
                response=resp,
                context={"page": page_no},
            )
        except Exception:
            pass
        return (resp.choices[0].message.content or "").strip()

    def _ocr_image_b64(self, image_b64: str, page_no: int) -> str:
        """Backwards-compatible wrapper — used by legacy code paths."""
        return self._ocr_page_with_retry(image_b64, page_no)

    def _ocr_page_with_retry(self, image_b64: str, page_no: int) -> str:
        """Call vision API with exponential backoff. Returns '' on final failure."""
        last_err = None
        for attempt in range(self.max_retries):
            try:
                text = self._call_vision_api(image_b64, page_no=page_no)
                logger.info(f"OCR page {page_no}: {len(text)} chars (attempt {attempt+1})")
                return text
            except Exception as e:
                last_err = e
                # 429 (rate limit) or transient — back off longer
                is_rate_limit = "429" in str(e) or "rate" in str(e).lower()
                wait = (2 ** attempt) * (2 if is_rate_limit else 1)
                if attempt < self.max_retries - 1:
                    logger.warning(
                        f"OCR page {page_no} attempt {attempt+1} failed ({e}). "
                        f"Retrying in {wait}s"
                    )
                    time.sleep(wait)
        logger.error(f"OCR page {page_no} failed after {self.max_retries} tries: {last_err}")
        return ""

    def ocr_pdf(self, pdf_path: str) -> List[Dict]:
        """OCR ทุกหน้าของ PDF แบบ parallel.
        Returns: [{"page": 1, "content": "..."}, ...]  (in page order)
        """
        pdf_path = str(pdf_path)
        logger.info(
            f"OpenRouter OCR: {pdf_path} · model={self.model} · dpi={self.dpi} "
            f"· workers={self.max_workers} · retries={self.max_retries}"
        )

        # Step 1: render all pages sequentially (fitz is not thread-safe)
        pages: List[Tuple[int, str]] = []  # (page_no, base64_png)
        with fitz.open(pdf_path) as doc:
            total = len(doc)
            for i, page in enumerate(doc, start=1):
                png_bytes = self._render_page(page)
                pages.append((i, base64.b64encode(png_bytes).decode("ascii")))
            logger.info(f"Rendered {total} pages at DPI={self.dpi}")

        # Step 2: OCR pages in parallel
        start = time.time()
        results: List[Optional[Dict]] = [None] * len(pages)
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers) as pool:
            future_to_idx = {
                pool.submit(self._ocr_page_with_retry, b64, page_no): idx
                for idx, (page_no, b64) in enumerate(pages)
            }
            done = 0
            for fut in concurrent.futures.as_completed(future_to_idx):
                idx = future_to_idx[fut]
                page_no = pages[idx][0]
                try:
                    text = fut.result()
                except Exception as e:
                    logger.error(f"OCR page {page_no} unrecoverable: {e}")
                    text = ""
                results[idx] = {"page": page_no, "content": text}
                done += 1
                logger.info(f"OCR progress: {done}/{len(pages)}")

        elapsed = time.time() - start
        avg = elapsed / max(len(pages), 1)
        logger.info(f"OCR done in {elapsed:.1f}s ({avg:.1f}s/page avg, {self.max_workers} workers)")

        return [r for r in results if r is not None]


def ocr_pdf_via_openrouter(pdf_path: str) -> List[Dict]:
    """Convenience wrapper — reads env vars for config."""
    ocr = OpenRouterVisionOCR()
    return ocr.ocr_pdf(pdf_path)
