from paddleocr import PPStructureV3
from PIL import Image
import pdf2image

pdf_path = "uploads/015_co-op_03_ใบสมัครงานสหกิจศึกษา_update.pdf"

pages = pdf2image.convert_from_path(pdf_path, dpi=300)

engine = PPStructureV3(layout=True)

for i, page in enumerate(pages):
    result = engine(page)

    for block in result:
        if block["type"] == "table":
            img = Image.fromarray(block["img"])
            img.save(f"table_page{i}.png")

print("DONE")
