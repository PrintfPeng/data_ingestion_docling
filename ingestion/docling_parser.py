# ingestion/docling_parser.py

import logging
from pathlib import Path
from typing import List, Optional, Tuple, Any, Dict
from datetime import datetime
import os

import pandas as pd
from PIL import Image

# Docling Imports
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions, TableStructureOptions

from .schema import IngestedDocument, TableBlock, TextBlock, DocumentMetadata

logger = logging.getLogger(__name__)

# -------------------------------------------------------------------
# Helper Function for Complexity Check
# -------------------------------------------------------------------
def is_complex_table(df: pd.DataFrame, sparsity_threshold=0.5) -> bool:
    if df.empty: return False
    total_cells = df.size
    if total_cells == 0: return False
    empty_cells = df.isna().sum().sum() + (df.astype(str).map(lambda x: x.strip() == '')).sum().sum()
    return bool((empty_cells / total_cells) > sparsity_threshold)

# -------------------------------------------------------------------
# MAIN PARSER
# -------------------------------------------------------------------
class DoclingParser:
    def __init__(self, config=None):
        self.config = config
        pipeline_options = PdfPipelineOptions()
        pipeline_options.do_ocr = True
        pipeline_options.do_table_structure = True
        pipeline_options.table_structure_options = TableStructureOptions(do_cell_matching=True)
        pipeline_options.generate_page_images = True 
        pipeline_options.images_scale = 2.0 

        self.converter = DocumentConverter(
            format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)}
        )

    def parse(self, file_path: str) -> IngestedDocument:
        logger.info(f"Starting Docling parse for: {file_path}")
        try:
            conv_res = self.converter.convert(file_path)
            doc = conv_res.document
            
            page_images = {}
            for page_no, page in doc.pages.items():
                if hasattr(page, 'image') and page.image:
                    if hasattr(page.image, 'pil_image'):
                        page_images[page_no] = page.image.pil_image
                    elif hasattr(page.image, 'image'):
                        page_images[page_no] = page.image.image
                    else:
                        page_images[page_no] = page.image

            doc_id = Path(file_path).stem
            
            # Extract Components
            text_blocks = self._process_text(doc, doc_id)
            table_blocks = self._process_tables(doc, page_images, doc_id)

            metadata = DocumentMetadata(
                doc_id=doc_id,
                file_name=Path(file_path).name,
                doc_type="generic",
                page_count=len(doc.pages),
                ingested_at=datetime.now().isoformat(),
                source="docling"
            )

            # [FIXED] IngestedDocument Instantiation
            return IngestedDocument(
                metadata=metadata,
                texts=text_blocks,
                tables=table_blocks,
                images=[]
            )

        except Exception as e:
            logger.error(f"Docling parsing failed: {e}")
            raise

    def _process_text(self, doc, doc_id: str) -> List[TextBlock]:
        blocks = []
        for i, (item, level) in enumerate(doc.iterate_items()):
            if item.__class__.__name__ == 'TextItem':
                if not item.text.strip(): continue
                page_no = item.prov[0].page_no if item.prov else 1
                bbox = item.prov[0].bbox.as_tuple() if item.prov else None
                blocks.append(TextBlock(
                    id=f"text_{i}",
                    doc_id=doc_id,
                    content=item.text,
                    page=page_no,
                    bbox=bbox,
                    extra={"role": "paragraph"}
                ))
        return blocks

    def _process_tables(self, doc, page_images: dict, doc_id: str) -> List[TableBlock]:
        blocks = []
        for i, table in enumerate(doc.tables):
            # Pass doc to avoid deprecation warnings
            df = table.export_to_dataframe(doc)
            md = table.export_to_markdown(doc)
            is_complex = is_complex_table(df)
            saved_image_path = None
            
            if is_complex and self.config and getattr(self.config, 'output_dir', None):
                try:
                    page_no = table.prov[0].page_no
                    page_img = page_images.get(page_no)
                    if page_img:
                        bbox = table.prov[0].bbox
                        
                        # Calculate Crop Box
                        # ต้องแน่ใจว่าพิกัดถูกต้องและไม่ติดลบ
                        l, r = min(bbox.l, bbox.r), max(bbox.l, bbox.r)
                        t, b = min(bbox.t, bbox.b), max(bbox.t, bbox.b)
                        
                        scale = 2.0 # Match images_scale
                        crop_box = (l * scale, t * scale, r * scale, b * scale)
                        
                        table_img = page_img.crop(crop_box)
                        
                        img_dir = os.path.join(self.config.output_dir, "images")
                        os.makedirs(img_dir, exist_ok=True)
                        file_name = f"complex_tbl_p{page_no}_{i}.png"
                        table_img.save(os.path.join(img_dir, file_name))
                        saved_image_path = f"images/{file_name}"
                        logger.info(f"   ⚠️ Complex Form detected. Saved image: {saved_image_path}")
                except Exception as e:
                    logger.warning(f"Failed to save table image: {e}")

            page_no = table.prov[0].page_no if table.prov else 1
            blocks.append(TableBlock(
                id=f"TBL_{i}",
                doc_id=doc_id,
                page=page_no,
                columns=[str(c) for c in df.columns], 
                rows=df.values.tolist(),
                markdown=md,
                image_path=saved_image_path,
                is_complex=bool(is_complex),
                source="docling",
                structured_available=bool(not df.empty),
                bbox=table.prov[0].bbox.as_tuple() if table.prov else None
            ))
        return blocks

# [MANDATORY] DoclingImageParser for backward compatibility
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
            format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)}
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
                            "file_name": image_filename, # แก้ key เป็น file_name
                            "page": page_no,
                            "bbox": bbox
                        })
            return saved_images
        except Exception as e:
            print(f"❌ [DoclingImageParser] Error: {e}")
            return []