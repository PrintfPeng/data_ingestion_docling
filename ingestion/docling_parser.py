# ingestion/docling_parser.py

import logging
from pathlib import Path
from typing import List, Optional, Tuple, Any, Dict
from datetime import datetime
import os

import pandas as pd
import cv2
import numpy as np
from PIL import Image

# Docling Imports
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions, TableStructureOptions

from .schema import (
    IngestedDocument,
    TableBlock,
    TextBlock,
    DocumentMetadata,
    ImageBlock,
)


logger = logging.getLogger(__name__)


# -------------------------------------------------------------------
# MAIN PARSER (Table Extractor)
# -------------------------------------------------------------------
class DoclingParser:
    def __init__(self, config=None):
        self.config = config

        pipeline_options = PdfPipelineOptions()
        pipeline_options.do_ocr = True
        pipeline_options.do_table_structure = True
        pipeline_options.table_structure_options = TableStructureOptions(
            do_cell_matching=True
        )
        pipeline_options.generate_page_images = True
        pipeline_options.images_scale = 3.0

        self.converter = DocumentConverter(
            format_options={
                InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)
            }
        )

    def parse(self, file_path: str) -> IngestedDocument:
        logger.info(f"Starting Docling parse for: {file_path}")
        try:
            conv_res = self.converter.convert(file_path)
            doc = conv_res.document

            page_images = {}
            for page_no, page in doc.pages.items():
                if hasattr(page, "image") and page.image:
                    if hasattr(page.image, "pil_image"):
                        page_images[page_no] = page.image.pil_image
                    elif hasattr(page.image, "image"):
                        page_images[page_no] = page.image.image
                    else:
                        page_images[page_no] = page.image

            doc_id = Path(file_path).stem

            text_blocks = self._process_text(doc, doc_id)

            # Prepare output dir
            output_dir = None
            if self.config and getattr(self.config, "output_dir", None):
                output_dir = self.config.output_dir
            else:
                output_dir = str(Path("ingested") / doc_id)

            table_blocks, table_images = self._process_tables(
                doc, page_images, doc_id, output_dir
            )

            metadata = DocumentMetadata(
                doc_id=doc_id,
                file_name=Path(file_path).name,
                doc_type="generic",
                page_count=len(doc.pages),
                ingested_at=datetime.now().isoformat(),
                source="docling",
            )

            return IngestedDocument(
                metadata=metadata,
                texts=text_blocks,
                tables=table_blocks,
                images=table_images,
            )

        except Exception as e:
            logger.error(f"Docling parsing failed: {e}")
            raise

    def _process_text(self, doc, doc_id: str) -> List[TextBlock]:
        blocks = []
        for i, (item, level) in enumerate(doc.iterate_items()):
            if item.__class__.__name__ == "TextItem":
                if not item.text.strip():
                    continue
                page_no = item.prov[0].page_no if item.prov else 1
                bbox = item.prov[0].bbox.as_tuple() if item.prov else None
                blocks.append(
                    TextBlock(
                        id=f"text_{i}",
                        doc_id=doc_id,
                        content=item.text,
                        page=page_no,
                        bbox=bbox,
                        extra={"role": "paragraph"},
                    )
                )
        return blocks
    
    def _refine_table_crop(self, region: Image.Image) -> Image.Image:
   
        img_np = np.array(region)
        gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)

        # Adaptive threshold (robust for forms)
        thresh = cv2.adaptiveThreshold(
            gray, 255,
            cv2.ADAPTIVE_THRESH_MEAN_C,
            cv2.THRESH_BINARY_INV,
            15, 5
        )

        h, w = thresh.shape

        # Dynamic kernel size
        horizontal_kernel_len = max(20, w // 25)
        vertical_kernel_len = max(20, h // 25)

        horizontal_kernel = cv2.getStructuringElement(
            cv2.MORPH_RECT, (horizontal_kernel_len, 1)
        )
        vertical_kernel = cv2.getStructuringElement(
            cv2.MORPH_RECT, (1, vertical_kernel_len)
        )

        horizontal_lines = cv2.morphologyEx(
            thresh, cv2.MORPH_OPEN, horizontal_kernel
        )

        vertical_lines = cv2.morphologyEx(
            thresh, cv2.MORPH_OPEN, vertical_kernel
        )

        table_mask = cv2.add(horizontal_lines, vertical_lines)

        contours, _ = cv2.findContours(
            table_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        if not contours:
            return region  # fallback

        # Filter small contours
        valid_contours = []
        min_area = (w * h) * 0.05

        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area > min_area:
                valid_contours.append(cnt)

        if not valid_contours:
            return region

        largest = max(valid_contours, key=cv2.contourArea)
        x, y, cw, ch = cv2.boundingRect(largest)

        # Safety clamp
        x_end = min(w, x + cw)
        y_end = min(h, y + ch)

        refined = img_np[y:y_end, x:x_end]

        if refined.size == 0:
            return region

        return Image.fromarray(refined)


    def _process_tables(
        self, doc, page_images: dict, doc_id: str, output_dir: str
    ) -> Tuple[List[TableBlock], List[ImageBlock]]:

        blocks = []
        img_blocks = []

        img_output_dir = os.path.join(output_dir, "images")
        os.makedirs(img_output_dir, exist_ok=True)

        SCALE = 3.0  # must match pipeline_options.images_scale

        for i, table in enumerate(doc.tables):

            df = table.export_to_dataframe(doc)
            md = table.export_to_markdown(doc)

            saved_image_path = None
            page_no = table.prov[0].page_no if table.prov else 1
            bbox_tuple = table.prov[0].bbox.as_tuple() if table.prov else None

            try:
                page_img = page_images.get(page_no)

                if page_img and table.prov and table.prov[0].bbox:

                    bbox = table.prov[0].bbox

                    img_w, img_h = page_img.size

                    # Scale bbox
                    l = bbox.l * SCALE
                    t = bbox.t * SCALE
                    r = bbox.r * SCALE
                    b = bbox.b * SCALE

                    x0 = min(l, r)
                    y0 = min(t, b)
                    x1 = max(l, r)
                    y1 = max(t, b)

                    # Aggressive padding (8%)
                    pad_ratio = 0.08
                    pad_x = int((x1 - x0) * pad_ratio)
                    pad_y = int((y1 - y0) * pad_ratio)

                    final_x0 = max(0, int(x0 - pad_x))
                    final_y0 = max(0, int(y0 - pad_y))
                    final_x1 = min(img_w, int(x1 + pad_x))
                    final_y1 = min(img_h, int(y1 + pad_y))

                    if (final_x1 - final_x0) > 50 and (final_y1 - final_y0) > 30:

                        initial_crop = page_img.crop(
                            (final_x0, final_y0, final_x1, final_y1)
                        )

                        # 🔥 refine using OpenCV
                        table_img = self._refine_table_crop(initial_crop)

                        filename = f"table_p{page_no:03d}_{i:03d}.png"
                        full_save_path = os.path.join(img_output_dir, filename)
                        table_img.save(full_save_path)

                        saved_image_path = f"images/{filename}"

                        img_blk = ImageBlock(
                            id=f"img_tbl_{i}",
                            doc_id=doc_id,
                            page=page_no,
                            file_path=saved_image_path,
                            caption=f"Table {i+1} extracted from page {page_no}",
                            bbox=bbox_tuple,
                            section="table",
                            category="table_image",
                            extra={"source": "docling_table"},
                        )

                        img_blocks.append(img_blk)

            except Exception as e:
                logger.warning(f"[DoclingParser] Table crop failed: {e}")

            blocks.append(
                TableBlock(
                    id=f"TBL_{i}",
                    doc_id=doc_id,
                    page=page_no,
                    columns=[str(c) for c in df.columns] if df is not None else [],
                    rows=df.values.tolist() if df is not None else [],
                    markdown=md,
                    image_path=saved_image_path,
                    is_complex=True,
                    source="docling",
                    structured_available=bool(df is not None and not df.empty),
                    bbox=bbox_tuple,
                )
            )

        return blocks, img_blocks

# -------------------------------------------------------------------
# IMAGE PARSER (General Image Extractor)
# -------------------------------------------------------------------
class DoclingImageParser:
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        pipeline_options = PdfPipelineOptions()
        pipeline_options.do_ocr = False
        pipeline_options.do_table_structure = False
        pipeline_options.generate_page_images = False
        pipeline_options.generate_picture_images = True
        pipeline_options.images_scale = 2.0

        self.converter = DocumentConverter(
            format_options={
                InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)
            }
        )

    def extract_images(self, pdf_path: str, output_dir: str) -> List[Dict[str, Any]]:
        file_path = Path(pdf_path).resolve()
        out_path = Path(output_dir)
        out_path.mkdir(parents=True, exist_ok=True)

        print(f"[DoclingImageParser] Scanning images in: {file_path.name} ...")
        try:
            conv_res = self.converter.convert(file_path)
            doc = conv_res.document
            saved_images = []
            if hasattr(doc, "pictures") and doc.pictures:
                print(f"[DoclingImageParser] Found {len(doc.pictures)} images.")
                for i, picture in enumerate(doc.pictures):
                    page_no = 1
                    bbox = None
                    if picture.prov and picture.prov[0]:
                        page_no = picture.prov[0].page_no
                        if hasattr(picture.prov[0], "bbox"):
                            bbox = picture.prov[0].bbox.as_tuple()

                    image_filename = f"img_p{page_no:03d}_{i+1:03d}.png"
                    image_save_path = out_path / image_filename

                    img_obj = picture.get_image(doc)
                    if img_obj:
                        img_obj.save(image_save_path, "PNG")

                        # [CRITICAL FIX] Convert Absolute Path to Relative Path
                        # จาก C:\Users\...\ingested\doc_id\img.png -> ingested/doc_id/img.png
                        try:
                            # หา Path สัมพัทธ์จากจุดที่รันโปรแกรม (Root)
                            rel_path = os.path.relpath(image_save_path, os.getcwd())
                            # เปลี่ยน Backslash (\) เป็น Slash (/) เพื่อให้ Web Browser เข้าใจ
                            rel_path = rel_path.replace("\\", "/")
                        except ValueError:
                            # กันพลาดกรณีข้าม Drive
                            rel_path = str(image_save_path)

                        saved_images.append(
                            {
                                "index": i,
                                "file_path": rel_path,  # <--- แก้ตรงนี้ครับ! ส่ง Path สั้นไป
                                "file_name": image_filename,
                                "page": page_no,
                                "bbox": bbox,
                            }
                        )
            return saved_images
        except Exception as e:
            print(f"❌ [DoclingImageParser] Error: {e}")
            return []
