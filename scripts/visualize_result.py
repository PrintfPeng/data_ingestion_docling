import fitz  # PyMuPDF
import json
from pathlib import Path
import argparse
import sys

# สีสำหรับวาด (R, G, B)
COLOR_TEXT = (0, 1, 0)    # Green
COLOR_TABLE = (1, 0, 0)   # Red
COLOR_IMAGE = (0, 0, 1)   # Blue

def draw_rects(page, items, color, width=1.5, label_prefix=""):
    """ฟังก์ชันช่วยวาดกรอบสี่เหลี่ยม"""
    for item in items:
        if item.get("page") != page.number + 1:
            continue
        
        bbox = item.get("bbox")
        if bbox:
            # bbox มาใน format [x0, y0, x1, y1]
            rect = fitz.Rect(bbox)
            page.draw_rect(rect, color=color, width=width)
            
            # เขียน Label เล็กๆ (ถ้ามี ID)
            if "id" in item:
                page.insert_text((rect.x0, rect.y0 - 2), f"{label_prefix}{item['id']}", color=color, fontsize=6)

def visualize_output(pdf_path: str, output_root: str = "ingested"):
    pdf_path = Path(pdf_path)
    doc_id = pdf_path.stem
    ingested_dir = Path(output_root) / doc_id
    
    if not ingested_dir.exists():
        print(f"❌ ไม่พบโฟลเดอร์ผลลัพธ์ที่: {ingested_dir}")
        print("   (รัน scripts/run_ingestion.py ก่อนนะครับ)")
        return

    # โหลดไฟล์ JSON ผลลัพธ์
    try:
        texts = json.loads((ingested_dir / "text_clean.json").read_text(encoding="utf-8"))
        tables = json.loads((ingested_dir / "table_normalized.json").read_text(encoding="utf-8"))
        images = json.loads((ingested_dir / "image.json").read_text(encoding="utf-8"))
    except FileNotFoundError as e:
        print(f"❌ ไฟล์ JSON ไม่ครบ: {e}")
        return

    # เปิด PDF ต้นฉบับ
    doc = fitz.open(pdf_path)
    
    # สร้างโฟลเดอร์เก็บภาพ Debug
    debug_dir = ingested_dir / "debug_visuals"
    debug_dir.mkdir(exist_ok=True)

    print(f"🎨 กำลังวาดภาพตรวจสอบลงใน: {debug_dir} ...")

    for page_index in range(len(doc)):
        page = doc[page_index]
        
        # 1. วาด Text (Green)
        draw_rects(page, texts, COLOR_TEXT, width=0.5)
        
        # 2. วาด Table (Red) - วาดทับ Text เพื่อดูว่า Table ครอบ Text ไหม
        draw_rects(page, tables, COLOR_TABLE, width=2, label_prefix="TBL:")
        
        # 3. วาด Image (Blue)
        draw_rects(page, images, COLOR_IMAGE, width=2, label_prefix="IMG:")

        # บันทึกเป็นภาพ PNG
        pix = page.get_pixmap(dpi=150)
        output_img = debug_dir / f"page_{page_index + 1:03d}.png"
        pix.save(output_img)
    
    print("✅ เรียบร้อย! เข้าไปดูรูปภาพได้เลยครับ")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("pdf_path", help="Path to original PDF file")
    args = parser.parse_args()
    
    visualize_output(args.pdf_path)