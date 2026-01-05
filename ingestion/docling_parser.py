import os
import time
import logging
from pathlib import Path
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions, TableStructureOptions

# --- ⚙️ CONFIGURATION ---
ENABLE_TABLE_RECOGNITION = False # ปิดเพื่อความเร็ว
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
        pipeline_options.do_table_structure = ENABLE_TABLE_RECOGNITION
        
        # Config ให้สร้างรูปภาพ
        pipeline_options.generate_page_images = True
        pipeline_options.generate_picture_images = True
        pipeline_options.images_scale = 2.0 

        self.converter = DocumentConverter(
            format_options={
                InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)
            }
        )

    def parse_file(self, file_path: str):
        start_time = time.time()
        file_path = str(Path(file_path).resolve())
        file_name = Path(file_path).name
        
        self.logger.info(f"Processing document: {file_name}")
        print(f"กำลังประมวลผล: {file_path}...")

        try:
            # 1. Convert
            conv_res = self.converter.convert(file_path)
            doc = conv_res.document
            
            # 2. Save Images
            saved_images = []
            if hasattr(doc, 'pictures') and doc.pictures:
                print(f"🖼️  Found {len(doc.pictures)} images. Saving...")
                for i, picture in enumerate(doc.pictures):
                    # ตั้งชื่อไฟล์รูป
                    image_filename = f"{Path(file_name).stem}_img_{i+1}.png"
                    image_save_path = os.path.join(self.image_dir, image_filename)
                    
                    # ดึงรูปและบันทึก
                    img_obj = picture.get_image(doc)
                    if img_obj:
                        img_obj.save(image_save_path, "PNG")
                        saved_images.append(image_save_path)
                        # ❌ ลบบรรทัด picture.common_metadata ออก เพื่อแก้ Error
            
            print(f"✅ Saved {len(saved_images)} images to {self.image_dir}")

            # 3. Export Markdown
            md_text = doc.export_to_markdown()
            output_path = os.path.join(self.output_dir, f"{Path(file_name).stem}_enriched.md")
            with open(output_path, "w", encoding="utf-8") as f:
                f.write(md_text)
                
            elapsed = time.time() - start_time
            print(f"ใช้เวลา: {elapsed:.2f} วินาที")

            return doc

        except Exception as e:
            self.logger.error(f"Error processing {file_name}: {e}")
            print(f"❌ Error: {e}")
            raise e