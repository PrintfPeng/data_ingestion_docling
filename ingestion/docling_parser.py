# ingestion/docling_parser.py

import os
import logging
from pathlib import Path
from typing import List, Tuple, Dict, Any
from datetime import datetime

import cv2
import numpy as np
from PIL import Image

# Docling
from docling.document_converter import (
    DocumentConverter,
    PdfFormatOption,
)
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import (
    PdfPipelineOptions,
    TableStructureOptions,
    EasyOcrOptions, # <--- ใช้ EasyOCR แทน! ติดตั้งง่ายกว่าล้านเท่า!
)

# Project Schema
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
        
        # 🚀 [CRITICAL FIX] เปลี่ยนมาใช้ EasyOCR แทน เพราะติดตั้งง่ายบน Windows
        pipeline_options.ocr_options = EasyOcrOptions(lang=["th", "en"]) # EasyOCR ใช้รหัสภาษา th, en
        
        pipeline_options.do_table_structure = True
        pipeline_options.table_structure_options = TableStructureOptions(
            do_cell_matching=True
        )
        pipeline_options.generate_page_images = True
        pipeline_options.images_scale = 3.0
        self.image_scale = 3.0

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
                # พยายามดึงรูปภาพจาก Property ต่างๆ
                img = None
                if hasattr(page, "image") and page.image:
                    if hasattr(page.image, "pil_image"):
                        img = page.image.pil_image
                    elif hasattr(page.image, "image"):
                        img = page.image.image
                    else:
                        img = page.image

                # ถ้าหาไม่เจอ หรือเป็น None ให้บังคับสร้างใหม่ (Force Render)
                if img is None:
                    try:
                        # บังคับ Render ที่ Scale 3.0
                        img = page.get_image(scale=3.0)
                    except Exception as e:
                        logger.warning(
                            f"Could not render image for page {page_no}: {e}"
                        )

                if img:
                    page_images[page_no] = img

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

    # -------------------------------------------------------------------#
    def _detect_table_regions(self, pil_image):
        # Detect all possible table regions from a page.
        # Returns list of (x0,y0,x1,y1)

        img = np.array(pil_image)

        # FIX RGB conversion
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

        gray = cv2.equalizeHist(gray)

        thresh = cv2.adaptiveThreshold(
            gray,
            255,
            cv2.ADAPTIVE_THRESH_MEAN_C,
            cv2.THRESH_BINARY_INV,
            15,
            5,
        )

        h, w = gray.shape

        # Dynamic kernel
        horizontal_len = max(40, w // 25)
        vertical_len = max(40, h // 25)

        horizontal_kernel = cv2.getStructuringElement(
            cv2.MORPH_RECT, (horizontal_len, 1)
        )
        vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, vertical_len))

        horizontal = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, horizontal_kernel)
        vertical = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, vertical_kernel)

        table_mask = cv2.add(horizontal, vertical)

        kernel = np.ones((5, 5), np.uint8)
        table_mask = cv2.dilate(table_mask, kernel, iterations=2)

        contours, _ = cv2.findContours(
            table_mask,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE,
        )

        regions = []

        for cnt in contours:
            x, y, cw, ch = cv2.boundingRect(cnt)
            area = cw * ch

            if cw > 200 and ch > 80 and area > (w * h * 0.01):
                regions.append((x, y, x + cw, y + ch))

        return regions

    # -------------------------------------------------------------------#

    def _match_best_region(self, regions, docling_bbox):

        if not regions:
            return None

        if not docling_bbox:
            return regions[0]

        dx0, dy0, dx1, dy1 = docling_bbox

        best_score = 0
        best_region = None

        for x0, y0, x1, y1 in regions:

            inter_x0 = max(x0, dx0)
            inter_y0 = max(y0, dy0)
            inter_x1 = min(x1, dx1)
            inter_y1 = min(y1, dy1)

            if inter_x1 <= inter_x0 or inter_y1 <= inter_y0:
                continue

            inter_area = (inter_x1 - inter_x0) * (inter_y1 - inter_y0)
            box_area = (x1 - x0) * (y1 - y0)

            score = inter_area / float(box_area)

            if score > best_score:
                best_score = score
                best_region = (x0, y0, x1, y1)

        return best_region

    # -------------------------------------------------------------------
    # แก้ไข: Table Cropping Logic (Smart Expansion) - ไม่ใช้ OpenCV แล้ว
    # -------------------------------------------------------------------
    def _process_tables(
        self,
        doc,
        page_images: dict,
        doc_id: str,
        output_dir: str,
    ) -> Tuple[List[TableBlock], List[ImageBlock]]:

        blocks = []
        img_blocks = []

        # สร้างโฟลเดอร์ images
        img_output_dir = os.path.join(output_dir, "images")
        os.makedirs(img_output_dir, exist_ok=True)

        for i, table in enumerate(doc.tables):
            df = table.export_to_dataframe(doc)
            md = table.export_to_markdown(doc)

            saved_image_path = None
            page_no = table.prov[0].page_no if table.prov else 1
            bbox_tuple = table.prov[0].bbox.as_tuple() if table.prov else None

            # [FIXED] สร้าง Caption จากเนื้อหาตาราง (Markdown)
            # ตัดให้เหลือแค่ 300 ตัวอักษรพอสังเขป เพื่อให้ AI เข้าใจบริบทตาราง
            table_content_preview = md[:300].replace("\n", " ")
            table_caption = f"Table {i+1}: {table_content_preview}..."

            # [THE REAL FIX] เลิกคำนวณพิกัดตัดรูปเอง ใช้คำสั่ง Native ของ Docling ดึงรูปตารางออกมาเลย
            try:
                img_obj = table.get_image(doc)
                if img_obj:
                    filename = f"table_p{page_no:03d}_{i:03d}.png"
                    full_path = os.path.join(img_output_dir, filename)

                    # เซฟรูปลงโฟลเดอร์
                    img_obj.save(full_path, "PNG")
                    saved_image_path = f"images/{filename}"

                    # สร้าง ImageBlock
                    img_blocks.append(
                        ImageBlock(
                            id=f"img_tbl_{i}",
                            doc_id=doc_id,
                            page=page_no,
                            file_path=saved_image_path,
                            caption=table_caption,  # [FIXED] ใส่เนื้อหาตารางลงไปใน Caption
                            bbox=bbox_tuple,
                            section="table",
                            category="table_image",
                            extra={
                                "source": "docling_native",
                                "markdown": md,  # เก็บ Markdown ไว้ใน extra ด้วยเผื่อใช้
                            },
                        )
                    )
            except Exception as e:
                logger.warning(f"Failed to extract table image natively: {e}")

            # สร้าง TableBlock เสมอ
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
                                "file_path": rel_path,  # ส่ง Path สั้นไป
                                "file_name": image_filename,
                                "page": page_no,
                                "bbox": bbox,
                            }
                        )
            return saved_images
        except Exception as e:
            print(f"❌ [DoclingImageParser] Error: {e}")
            return []
