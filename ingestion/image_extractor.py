from __future__ import annotations

"""
image_extractor.py

หน้าที่:
- เปิดไฟล์ PDF
- ดึงรูปภาพทุกภาพในเอกสาร
- ระบุตำแหน่ง (BBox) ของรูปภาพบนหน้ากระดาษ
- บันทึกลงโฟลเดอร์
- แปลงผลลัพธ์เป็น list[ImageBlock]
"""

from pathlib import Path
from typing import List

import fitz  # PyMuPDF

from .schema import ImageBlock, BBox

def extract_images(
    file_path: str | Path,
    doc_id: str,
    output_root: str | Path = "ingested",
) -> List[ImageBlock]:
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"PDF file not found: {path}")

    output_root = Path(output_root)
    image_dir = output_root / doc_id / "images"
    image_dir.mkdir(parents=True, exist_ok=True)

    pdf_doc = fitz.open(path)
    image_blocks: List[ImageBlock] = []

    try:
        image_counter = 0

        # loop ทุกหน้า
        for page_index in range(pdf_doc.page_count):
            page = pdf_doc[page_index]
            page_number = page_index + 1

            # get_images(full=True) คืน list ของ image objects ในหน้านั้น
            images = page.get_images(full=True)

            for img_index, img in enumerate(images, start=1):
                xref = img[0]  # image reference id ใน PDF
                
                # --- Logic ใหม่: หาตำแหน่ง (BBox) ของรูปภาพ ---
                rects = page.get_image_rects(xref)
                if not rects:
                    continue # ข้ามถ้าระบุตำแหน่งไม่ได้

                # เลือก BBox แรก (ส่วนใหญ่รูป 1 xref จะวางที่เดียว)
                r = rects[0]
                bbox: BBox = (
                    round(r.x0, 2), 
                    round(r.y0, 2), 
                    round(r.x1, 2), 
                    round(r.y1, 2)
                )

                # Extract ข้อมูลรูปภาพ
                try:
                    base_image = pdf_doc.extract_image(xref)
                except Exception as e:
                    print(f"[image_extractor] Error extracting image xref={xref}: {e}")
                    continue

                img_bytes: bytes = base_image["image"]
                img_ext: str = base_image.get("ext", "png")
                width: int = base_image.get("width", 0)
                height: int = base_image.get("height", 0)

                # กรองรูปที่เล็กเกินไป (เช่น icon, เส้นคั่น)
                if width < 100 or height < 100:
                    continue

                image_counter += 1
                img_id = f"img_{image_counter:04d}"

                filename = f"img_p{page_number:03d}_{img_index:03d}.{img_ext}"
                file_path_on_disk = image_dir / filename

                # เซฟรูปลงดิสก์
                with open(file_path_on_disk, "wb") as f:
                    f.write(img_bytes)

                image_block = ImageBlock(
                    id=img_id,
                    doc_id=doc_id,
                    page=page_number,
                    file_path=str(file_path_on_disk),
                    caption=None,
                    section=None,
                    category=None,
                    bbox=bbox,  # มีค่า bbox แล้ว
                    extra={
                        "width": width,
                        "height": height,
                        "xref": xref,
                    },
                )
                image_blocks.append(image_block)

        return image_blocks

    finally:
        pdf_doc.close()

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("pdf_path")
    parser.add_argument("--doc-id")
    args = parser.parse_args()
    
    doc_id = args.doc_id or Path(args.pdf_path).stem
    imgs = extract_images(args.pdf_path, doc_id=doc_id)
    print(f"Extracted {len(imgs)} images")