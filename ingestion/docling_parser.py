import os
from typing import List, Optional
from pathlib import Path

# Docling Core & Main imports
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import (
    PdfPipelineOptions,
    PictureDescriptionVlmOptions,
)
from docling_core.transforms.serializer.markdown import MarkdownDocSerializer

class DoclingParser:
    def __init__(self, output_dir: str = "ingested"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # 1. ตั้งค่า Pipeline สำหรับ PDF และ OCR ภาษาไทย
        pipeline_options = PdfPipelineOptions(
            do_ocr=True,
            do_picture_description=True, # เปิด Semantic Enrichment สำหรับรูปภาพ
            generate_picture_images=True, # บันทึกไฟล์รูปภาพแยกออกมา
        )
        
        # ตั้งค่า VLM สำหรับอธิบายรูปภาพ (สามารถเปลี่ยนโมเดลได้ตามความเหมาะสม)
        pipeline_options.picture_description_options = PictureDescriptionVlmOptions(
            repo_id="HuggingFaceTB/SmolVLM-256M-Instruct",
            prompt="Describe this picture in Thai. Be precise and concise.",
        )

        # 2. สร้าง Converter พร้อมรองรับภาษาไทย
        self.converter = DocumentConverter(
            format_options={
                InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)
            }
        )

    def parse_file(self, file_path: str):
        """แปลงไฟล์ PDF เป็น Markdown พร้อมคำบรรยายรูปภาพ"""
        print(f"กำลังประมวลผล: {file_path}...")
        
        # เริ่มการแปลงเอกสาร
        result = self.converter.convert(source=file_path)
        doc = result.document
        
        # 3. ใช้ MarkdownDocSerializer เพื่อดึงเนื้อหาออกมา
        # เนื้อหานี้จะรวมทั้งข้อความ ตาราง (ในรูปแบบ Markdown) และ Placeholder ของรูปภาพ
        serializer = MarkdownDocSerializer(doc=doc)
        ser_result = serializer.serialize()
        
        # บันทึกผลลัพธ์เป็นไฟล์ .md
        file_name = Path(file_path).stem
        output_path = self.output_dir / f"{file_name}_enriched.md"
        
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(ser_result.text)
            
        print(f"สำเร็จ! บันทึกไฟล์ที่: {output_path}")
        return ser_result.text

# ตัวอย่างการใช้งาน
if __name__ == "__main__":
    parser = DoclingParser()
    # ทดสอบกับไฟล์คู่มือ SHARP ของคุณ
    # parser.parse_file("path/to/TINS-B784CBRZ (CHANG).pdf")