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
            rect = fitz.Rect(bbox)
            page.draw_rect(rect, color=color, width=width)
            if "id" in item:
                page.insert_text((rect.x0, rect.y0 - 2), f"{label_prefix}{item['id']}", color=color, fontsize=6)

def visualize_output(pdf_path: str, doc_id: str = None, output_root: str = "ingested"):
    pdf_path = Path(pdf_path)
    
    # ถ้าไม่ได้ส่ง doc_id มา ให้ใช้ชื่อไฟล์เป็น default
    if not doc_id:
        doc_id = pdf_path.stem

    ingested_dir = Path(output_root) / doc_id
    
    if not ingested_dir.exists():
        print(f"❌ ไม่พบโฟลเดอร์ผลลัพธ์ที่: {ingested_dir}")
        print(f"   (เช็คว่า Doc ID ตรงกับตอน run_all หรือไม่: '{doc_id}')")
        return

    # พยายามโหลดไฟล์ JSON (รองรับชื่อไฟล์หลายแบบ)
    try:
        # ลองโหลด text_clean.json ก่อน (ปกติสร้างจาก cleaning pipeline)
        text_path = ingested_dir / "text_clean.json"
        if not text_path.exists():
            text_path = ingested_dir / "text.json" # fallback

        # ลองโหลด table.json หรือ table_normalized.json
        table_path = ingested_dir / "table_normalized.json"
        if not table_path.exists():
            table_path = ingested_dir / "table.json" # fallback

        texts = json.loads(text_path.read_text(encoding="utf-8"))
        tables = json.loads(table_path.read_text(encoding="utf-8"))
        images = json.loads((ingested_dir / "image.json").read_text(encoding="utf-8"))
    except Exception as e:
        print(f"❌ โหลดไฟล์ JSON ไม่สำเร็จ: {e}")
        return

    doc = fitz.open(pdf_path)
    debug_dir = ingested_dir / "debug_visuals"
    debug_dir.mkdir(exist_ok=True)

    print(f"🎨 วาดภาพตรวจสอบลงใน: {debug_dir} (DocID: {doc_id})")

    for page_index in range(len(doc)):
        page = doc[page_index]
        draw_rects(page, texts, COLOR_TEXT, width=0.5)
        draw_rects(page, tables, COLOR_TABLE, width=2, label_prefix="TBL:")
        draw_rects(page, images, COLOR_IMAGE, width=2, label_prefix="IMG:")

        output_img = debug_dir / f"page_{page_index + 1:03d}.png"
        page.get_pixmap(dpi=150).save(output_img)
    
    print("✅ เรียบร้อย!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("pdf_path", help="Path to original PDF file")
    # เพิ่ม argument นี้เพื่อให้รับค่า --doc-id ได้
    parser.add_argument("--doc-id", default=None, help="Document ID used during ingestion") 
    args = parser.parse_args()
    
    visualize_output(args.pdf_path, doc_id=args.doc_id)