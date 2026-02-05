# ingestion/docling_parser.py

import logging
from pathlib import Path
from typing import List, Optional, Tuple, Any, Dict

import pandas as pd
# Docling Imports
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions, TableStructureOptions
from docling.datamodel.document import TableItem

# Image Processing
from PIL import Image
import os

from .schema import IngestedDocument, TableBlock, TextBlock

logger = logging.getLogger(__name__)

# -------------------------------------------------------------------
# Helper Function for Complexity Check
# -------------------------------------------------------------------
def is_complex_table(df: pd.DataFrame, sparsity_threshold=0.5) -> bool:
    """
    Checks if a table is 'complex' (likely a form) based on sparsity.
    If empty cells > 50%, it's considered complex.
    """
    if df.empty:
        return False
    
    total_cells = df.size
    if total_cells == 0:
        return False
        
    # Count empty cells (NaN, None, or Empty String)
    empty_cells = df.isna().sum().sum() + (df.astype(str).map(lambda x: x.strip() == '')).sum().sum()
    
    sparsity = empty_cells / total_cells
    return sparsity > sparsity_threshold


# -------------------------------------------------------------------
# MAIN PARSER (Text & Tables & Hybrid Ingestion)
# -------------------------------------------------------------------
class DoclingParser:
    """
    Parser using IBM Docling for deep structural analysis of PDFs.
    Extracts text blocks and high-quality tables.
    """

    def __init__(self, config=None):
        self.config = config
        
        # Configure Docling Options
        pipeline_options = PdfPipelineOptions()
        pipeline_options.do_ocr = True  # Enable OCR
        pipeline_options.do_table_structure = True
        pipeline_options.table_structure_options = TableStructureOptions(
            do_cell_matching=True
        )
        # [IMPORTANT] Enable image generation
        pipeline_options.generate_page_images = True 
        pipeline_options.images_scale = 2.0 

        self.converter = DocumentConverter(
            format_options={
                InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)
            }
        )

    def parse(self, file_path: str) -> IngestedDocument:
        logger.info(f"Starting Docling parse for: {file_path}")
        
        try:
            # 1. Convert Document
            conv_res = self.converter.convert(file_path)
            doc = conv_res.document
            
            # [FIXED] Extract Page Images Correctly for Docling v2
            page_images = {}
            for page_no, page in doc.pages.items():
                try:
                    if hasattr(page, 'image') and page.image:
                        if hasattr(page.image, 'pil_image'):
                            page_images[page_no] = page.image.pil_image
                        elif hasattr(page.image, 'image'):
                            page_images[page_no] = page.image.image
                        elif isinstance(page.image, Image.Image):
                            page_images[page_no] = page.image
                        else:
                             logger.warning(f"Unknown image type for page {page_no}: {type(page.image)}")
                except Exception as img_err:
                    logger.warning(f"Could not extract image for page {page_no}: {img_err}")
            
            # 2. Extract Components
            text_blocks = self._process_text(doc)
            table_blocks = self._process_tables(doc, page_images)

            return IngestedDocument(
                doc_id=Path(file_path).name,
                text_blocks=text_blocks,
                table_blocks=table_blocks,
                metadata={"engine": "docling", "page_count": len(doc.pages)}
            )

        except Exception as e:
            logger.error(f"Docling parsing failed: {e}")
            raise

    def _process_text(self, doc) -> List[TextBlock]:
        blocks = []
        i = 0
        for item, level in doc.iterate_items():
            if item.__class__.__name__ == 'TextItem':
                if not item.text.strip():
                    continue

                page_no = item.prov[0].page_no if item.prov else 1
                bbox = item.prov[0].bbox.as_tuple() if item.prov else None
                
                # [CRITICAL FIX] เปลี่ยนจาก text=... เป็น content=... และเพิ่ม id
                blocks.append(TextBlock(
                    id=f"text_{i}",
                    doc_id="unknown",
                    content=item.text,  # <--- แก้ตรงนี้ครับ
                    page=page_no,
                    bbox=bbox,
                    extra={"role": "paragraph"}
                ))
                i += 1
        return blocks

    def _process_tables(self, doc, page_images: dict) -> List[TableBlock]:
        blocks = []
        
        for i, table in enumerate(doc.tables):
            # 1. Export Data
            df = table.export_to_dataframe()
            markdown_content = table.export_to_markdown()
            
            # 2. Complexity Check
            is_complex = is_complex_table(df)
            saved_image_path = None
            
            # 3. Handle Image Saving
            if is_complex and self.config and getattr(self.config, 'output_dir', None):
                try:
                    page_no = table.prov[0].page_no
                    page_img = page_images.get(page_no)
                    
                    if page_img:
                        bbox = table.prov[0].bbox
                        # Crop logic
                        crop_box = (bbox.l, bbox.t, bbox.r, bbox.b)
                        table_img = page_img.crop(crop_box)
                        
                        filename = f"complex_tbl_p{page_no}_{i}.png"
                        images_dir = os.path.join(self.config.output_dir, "images")
                        os.makedirs(images_dir, exist_ok=True)
                        
                        output_path = os.path.join(images_dir, filename)
                        table_img.save(output_path)
                        
                        saved_image_path = f"images/{filename}"
                        logger.info(f"   ⚠️ Complex Form detected. Saved image: {saved_image_path}")
                        
                except Exception as e:
                    logger.warning(f"   ❌ Failed to save complex table image: {e}")

            # 4. Create Block
            page_no = table.prov[0].page_no if table.prov else 1
            bbox = table.prov[0].bbox.as_tuple() if table.prov else None

            table_block = TableBlock(
                id=f"TBL_{i}",
                doc_id="unknown",
                page=page_no,
                bbox=bbox,
                columns=df.columns.tolist(),
                rows=df.values.tolist(),
                markdown=markdown_content,
                image_path=saved_image_path,
                is_complex=is_complex,
                source="docling",
                structured_available=not df.empty
            )
            
            blocks.append(table_block)
            
        return blocks


# -------------------------------------------------------------------
# IMAGE PARSER (Legacy support)
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
            
            if hasattr(doc, 'pictures') and doc.pictures:
                print(f"[DoclingImageParser] Found {len(doc.pictures)} images.")
                
                for i, picture in enumerate(doc.pictures):
                    page_no = 1
                    bbox = None
                    if picture.prov and picture.prov[0]:
                        page_no = picture.prov[0].page_no
                        if hasattr(picture.prov[0], 'bbox'):
                            bbox = picture.prov[0].bbox.as_tuple()
                    
                    image_filename = f"img_p{page_no:03d}_{i+1:03d}.png"
                    image_save_path = out_path / image_filename
                    
                    img_obj = picture.get_image(doc)
                    if img_obj:
                        img_obj.save(image_save_path, "PNG")
                        
                        saved_images.append({
                            "index": i,
                            "file_path": str(image_save_path),
                            "filename": image_filename,
                            "page": page_no,
                            "bbox": bbox
                        })
            else:
                print("[DoclingImageParser] No images found in this document.")
            
            return saved_images

        except Exception as e:
            print(f"❌ [DoclingImageParser] Error: {e}")
            return []