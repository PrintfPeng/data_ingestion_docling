import sys
import os
import glob
from pathlib import Path

# Fix path import
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from ingestion.docling_parser import DoclingParser
from backend.services.chunking import MarkdownChunker
from backend.services.vector_store import VectorStore
# ✅ เพิ่ม Image Embedder
from backend.services.image_embedder import ImageEmbedder

def run_pipeline(pdf_path: str):
    # 1. Setup
    if not os.path.exists(pdf_path):
        print(f"Error: File not found at {pdf_path}")
        return

    print(f"--- 🚀 Starting Ingestion for: {Path(pdf_path).name} ---")
    
    # 2. Docling Processing (OCR + VLM + Markdown + Image Extraction)
    # หมายเหตุ: ต้องมั่นใจว่า DoclingParser ในไฟล์ docling_parser.py มีการรับค่า image_dir และ return doc object
    parser = DoclingParser(output_dir="ingested_md", image_dir="ingested_images")
    
    # --- Logic ตรวจสอบ Type ของ Docling (Smart Check) ---
    raw_result = parser.parse_file(pdf_path)
    
    doc = None
    md_text = ""

    def is_docling_object(obj):
        # เช็คว่าเป็น Object Docling จริงหรือไม่ (ต้องมี export_to_markdown)
        return hasattr(obj, 'export_to_markdown') and not isinstance(obj, str)

    # กรณี 1: Return มาเป็น List/Tuple
    if isinstance(raw_result, (list, tuple)):
        for item in raw_result:
            if isinstance(item, str):
                md_text = item
            elif is_docling_object(item):
                doc = item
    
    # กรณี 2: Return มาเป็น Object ตัวเดียว
    elif is_docling_object(raw_result):
        doc = raw_result

    # Safety Check
    if doc is None or isinstance(doc, str):
        print("❌ CRITICAL ERROR: ไม่สามารถหา DoclingDocument Object ได้")
        print(f"   สิ่งที่ได้รับมาคือ type: {type(raw_result)}")
        return 

    print(f"DEBUG: doc type={type(doc)} (Correct Object), md_text length={len(md_text)}")
    # -----------------------------------------------------
    
    # 3. Text Chunking
    print("--- ✂️ Chunking & Normalizing Text ---")
    chunker = MarkdownChunker()
    chunks = chunker.create_chunks(doc)
    
    # 4. Save Text to Vector DB
    print(f"--- 💾 Saving {len(chunks)} text chunks to ChromaDB ---")
    vector_store = VectorStore()
    
    ids = [f"{Path(pdf_path).stem}_{c['metadata']['chunk_id']}" for c in chunks]
    documents = [c["content"] for c in chunks]
    metadatas = [c["metadata"] for c in chunks]
    
    for m in metadatas:
        m["file_name"] = Path(pdf_path).name
        m["type"] = "text" # ระบุว่าเป็น Text

    # บันทึก Text ลง Collection หลัก
    vector_store.add_documents(ids=ids, documents=documents, metadatas=metadatas)
    
    # 5. Image Processing (ส่วนใหม่) 🖼️
    print("--- 🖼️ Processing Images ---")
    file_stem = Path(pdf_path).stem
    image_dir = "ingested_images"
    
    # ค้นหารูปภาพทั้งหมดที่เกี่ยวข้องกับ PDF นี้
    image_paths = glob.glob(os.path.join(image_dir, f"{file_stem}_img_*.png"))
    
    if image_paths:
        print(f"Found {len(image_paths)} images. Starting embedding...")
        embedder = ImageEmbedder() # เรียกใช้โมเดล CLIP/SigLIP
        
        image_vectors = []
        image_metadatas = []
        image_ids = []
        
        for img_path in image_paths:
            print(f"  - Embedding: {Path(img_path).name}")
            vector = embedder.embed_image(img_path)
            
            if vector:
                img_name = Path(img_path).name
                image_ids.append(img_name)
                image_vectors.append(vector)
                image_metadatas.append({
                    "file_name": Path(pdf_path).name,
                    "source_doc": file_stem,
                    "type": "image",
                    "image_path": img_path
                })

        print(f"✅ Generated vectors for {len(image_vectors)} images")
        
        # --- หมายเหตุสำคัญสำหรับการบันทึกลง DB ---
        # เนื่องจาก Vector ของรูปภาพ (เช่น CLIP 512 dim) กับ Text (เช่น BGE-M3 1024 dim) ขนาดไม่เท่ากัน
        # คุณต้องมีฟังก์ชัน add_images ใน vector_store.py ที่บันทึกลง Collection แยกต่างหาก (เช่น 'image_collection')
        
        try:
            if hasattr(vector_store, 'add_images'):
                print(f"--- 💾 Saving {len(image_vectors)} images to ChromaDB (Image Collection) ---")
                vector_store.add_images(ids=image_ids, embeddings=image_vectors, metadatas=image_metadatas)
            else:
                print("⚠️ Warning: ยังไม่มีฟังก์ชัน 'add_images' ใน VectorStore ข้ามการบันทึกรูปภาพ")
                print("   (แต่ไฟล์รูปภาพและ Vector ถูกสร้างเรียบร้อยแล้ว)")
        except Exception as e:
            print(f"❌ Error saving images: {e}")

    else:
        print("No images found for this document.")

    print("--- ✅ Ingestion Complete! Ready for RAG ---")

if __name__ == "__main__":
    target_pdf = sys.argv[1] if len(sys.argv) > 1 else "TINS-B784CBRZ (CHANG).pdf"
    # target_pdf = r"C:\Users\ASUS\Desktop\test_flowDG.pdf"
    run_pipeline(target_pdf)