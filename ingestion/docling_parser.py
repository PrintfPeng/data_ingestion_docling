import os
import time
import logging
from pathlib import Path
from typing import List, Dict, Any

from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions, TableStructureOptions

# --- CONFIG ---
ENABLE_OCR = True  

class DoclingParser:
    def __init__(self, output_dir="ingested_md", image_dir="ingested_images"):
        self.output_dir = output_dir
        self.image_dir = image_dir 
        
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(self.image_dir, exist_ok=True) 
        
        self.logger = logging.getLogger(__name__)
        
        # ตั้งค่า Pipeline
        pipeline_options = PdfPipelineOptions()
        pipeline_options.do_ocr = ENABLE_OCR
        pipeline_options.do_table_structure = True
        
        # Config ให้สร้างรูปภาพ
        pipeline_options.generate_page_images = True
        pipeline_options.generate_picture_images = True
        pipeline_options.images_scale = 2.0 

        self.converter = DocumentConverter(
            format_options={
                InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)
            }
        )

    def parse_file(self, file_path: str) -> Dict[str, Any]:
        """
        Return dict ที่มี:
        - doc: Docling Document Object
        - saved_images: List ของ dict {reference, path, page}
        - markdown: string content
        """
        start_time = time.time()
        file_path = str(Path(file_path).resolve())
        file_name = Path(file_path).name
        
        self.logger.info(f"Processing document with Docling: {file_name}")
        print(f"[Docling] Parsing: {file_name} ...")

        try:
            # 1. Convert
            conv_res = self.converter.convert(file_path)
            doc = conv_res.document
            
            # 2. Save Images & Collect Metadata
            saved_images = []
            if hasattr(doc, 'pictures') and doc.pictures:
                print(f"[Docling] Found {len(doc.pictures)} images. Saving...")
                for i, picture in enumerate(doc.pictures):
                    # ตั้งชื่อไฟล์รูป
                    image_filename = f"{Path(file_name).stem}_img_{i+1}.png"
                    image_save_path = os.path.join(self.image_dir, image_filename)
                    
                    # ดึงรูปและบันทึก
                    img_obj = picture.get_image(doc)
                    if img_obj:
                        img_obj.save(image_save_path, "PNG")
                        
                        # เก็บข้อมูลไว้ return
                        saved_images.append({
                            "index": i,
                            "path": image_save_path,
                            "page": picture.prov[0].page_no if picture.prov else 1,
                            "bbox": picture.prov[0].bbox.as_tuple() if picture.prov else None,
                            "obj": picture # เก็บ object ไว้เผื่อใช้
                        })
            
            # 3. Export Markdown (Optional: save to disk)
            md_text = doc.export_to_markdown()
            # output_path = os.path.join(self.output_dir, f"{Path(file_name).stem}_enriched.md")
            # with open(output_path, "w", encoding="utf-8") as f:
            #     f.write(md_text)
                
            elapsed = time.time() - start_time
            print(f"[Docling] Finished in {elapsed:.2f}s")

            return {
                "doc": doc,
                "saved_images": saved_images,
                "markdown": md_text
            }

        except Exception as e:
            self.logger.error(f"Error processing {file_name}: {e}")
            print(f"❌ Docling Error: {e}")
            raise e