import time
import logging
from pathlib import Path
from typing import List, Dict, Any

# ต้องติดตั้ง docling ก่อน (pip install docling)
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions

# ตั้งค่า OCR (เปิด True ไว้จะช่วยให้ระบุตำแหน่งภาพในหน้าที่มีข้อความทับได้แม่นยำขึ้น)
ENABLE_OCR = True  

class DoclingImageParser:
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
        # ตั้งค่า Pipeline ของ Docling ให้โฟกัสแค่ "รูปภาพ"
        pipeline_options = PdfPipelineOptions()
        pipeline_options.do_ocr = ENABLE_OCR
        pipeline_options.do_table_structure = False # ปิด เพราะเรามี table_extractor แยกแล้ว (ประหยัดเวลา)
        pipeline_options.generate_page_images = False # ปิด ไม่เอาภาพ Screenshot ทั้งหน้า
        pipeline_options.generate_picture_images = True # ✅ เปิด เอาเฉพาะภาพประกอบ (Bitmap/Figure)
        pipeline_options.images_scale = 2.0 # คุณภาพสูง (2x) เพื่อให้ AI อ่านชัด

        self.converter = DocumentConverter(
            format_options={
                InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)
            }
        )

    def extract_images(self, pdf_path: str, output_dir: str) -> List[Dict[str, Any]]:
        """
        หน้าที่: อ่าน PDF ด้วย Docling -> ดึงรูป -> เซฟลง output_dir
        คืนค่า: List ของ Dict ข้อมูลรูปภาพ (file_path, filename, page, bbox)
        """
        start_time = time.time()
        file_path = Path(pdf_path).resolve()
        
        # สร้างโฟลเดอร์ปลายทางถ้ายังไม่มี
        out_path = Path(output_dir)
        out_path.mkdir(parents=True, exist_ok=True)
        
        print(f"[Docling] Parsing images from: {file_path.name} ...")

        try:
            # เริ่มกระบวนการแปลง (Convert)
            conv_res = self.converter.convert(file_path)
            doc = conv_res.document
            
            saved_images = []
            
            # ตรวจสอบว่าเจอรูปไหม (Docling เก็บรูปไว้ใน doc.pictures)
            if hasattr(doc, 'pictures') and doc.pictures:
                print(f"[Docling] Found {len(doc.pictures)} images. Saving...")
                
                for i, picture in enumerate(doc.pictures):
                    # หาเลขหน้า (Docling เริ่มนับหน้า 1)
                    page_no = 1
                    bbox = None
                    
                    # พยายามดึงข้อมูลตำแหน่ง (Provenance)
                    if picture.prov and picture.prov[0]:
                        page_no = picture.prov[0].page_no
                        # แปลง BBox เป็น tuple (x0, y0, x1, y1)
                        if hasattr(picture.prov[0], 'bbox'):
                            bbox = picture.prov[0].bbox.as_tuple()
                    
                    # ตั้งชื่อไฟล์ให้เป็นระเบียบ: img_p{หน้า}_{ลำดับ}.png
                    image_filename = f"img_p{page_no:03d}_{i+1:03d}.png"
                    image_save_path = out_path / image_filename
                    
                    # ดึง Object ภาพออกมาและบันทึก
                    img_obj = picture.get_image(doc)
                    if img_obj:
                        img_obj.save(image_save_path, "PNG")
                        
                        saved_images.append({
                            "index": i,
                            "file_path": str(image_save_path), # Path เต็ม
                            "filename": image_filename,        # ชื่อไฟล์
                            "page": page_no,
                            "bbox": bbox
                        })
            else:
                print("[Docling] No images found in this document.")
            
            print(f"[Docling] Finished extracting images in {time.time() - start_time:.2f}s")
            return saved_images

        except Exception as e:
            print(f"❌ [Docling] Error: {e}")
            return [] # คืนค่าว่าง เพื่อไม่ให้ Pipeline หลักพัง