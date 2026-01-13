# ingestion/docling_parser.py

import os
import time
import logging
from pathlib import Path
from typing import List, Dict, Any

from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions

ENABLE_OCR = True  

class DoclingParser:
    def __init__(self, output_dir="ingested_md", image_dir="ingested_images"):
        self.output_dir = output_dir
        self.image_dir = image_dir 
        
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(self.image_dir, exist_ok=True) 
        
        self.logger = logging.getLogger(__name__)
        
        pipeline_options = PdfPipelineOptions()
        pipeline_options.do_ocr = ENABLE_OCR
        pipeline_options.do_table_structure = True
        pipeline_options.generate_page_images = True
        pipeline_options.generate_picture_images = True
        pipeline_options.images_scale = 2.0 

        self.converter = DocumentConverter(
            format_options={
                InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)
            }
        )

    def parse_file(self, file_path: str) -> Dict[str, Any]:
        start_time = time.time()
        file_path = str(Path(file_path).resolve())
        file_name = Path(file_path).name
        
        print(f"[Docling] Parsing: {file_name} ...")

        try:
            conv_res = self.converter.convert(file_path)
            doc = conv_res.document
            
            saved_images = []
            if hasattr(doc, 'pictures') and doc.pictures:
                print(f"[Docling] Found {len(doc.pictures)} images. Saving...")
                for i, picture in enumerate(doc.pictures):
                    
                    # [FIX] ดึงเลขหน้า และตั้งชื่อไฟล์ให้มีคำว่า _page_{no}_
                    page_no = 1
                    if picture.prov and picture.prov[0]:
                        page_no = picture.prov[0].page_no
                    
                    # ชื่อไฟล์แบบใหม่: docname_page_1_img_1.png
                    image_filename = f"{Path(file_name).stem}_page_{page_no}_img_{i+1}.png"
                    image_save_path = os.path.join(self.image_dir, image_filename)
                    
                    img_obj = picture.get_image(doc)
                    if img_obj:
                        img_obj.save(image_save_path, "PNG")
                        saved_images.append({
                            "index": i,
                            "path": image_save_path,
                            "page": page_no,
                            "bbox": picture.prov[0].bbox.as_tuple() if picture.prov else None
                        })
            
            md_text = doc.export_to_markdown()
            print(f"[Docling] Finished in {time.time() - start_time:.2f}s")

            return {"doc": doc, "saved_images": saved_images, "markdown": md_text}

        except Exception as e:
            print(f"❌ Docling Error: {e}")
            raise e