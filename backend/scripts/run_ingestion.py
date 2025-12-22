import sys
import os
from pathlib import Path

# Fix path import
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from ingestion.docling_parser import DoclingParser
from backend.services.chunking import MarkdownChunker
from backend.services.vector_store import VectorStore

def run_pipeline(pdf_path: str):
    # 1. Setup
    if not os.path.exists(pdf_path):
        print(f"Error: File not found at {pdf_path}")
        return

    print(f"--- 🚀 Starting Ingestion for: {Path(pdf_path).name} ---")
    
    # 2. Docling Processing (OCR + VLM + Markdown)
    parser = DoclingParser(output_dir="ingested_md")
    doc, md_text = parser.parse_file(pdf_path) # แก้ให้รับค่า return 2 ตัว (doc object และ text)
    
    # 3. Chunking (พร้อมแก้สระภาษาไทยในตัว)
    print("--- ✂️ Chunking & Normalizing Text ---")
    chunker = MarkdownChunker()
    chunks = chunker.create_chunks(doc)
    
    # 4. Save to Vector DB
    print(f"--- 💾 Saving {len(chunks)} chunks to ChromaDB ---")
    vector_store = VectorStore()
    
    ids = [f"{Path(pdf_path).stem}_{c['metadata']['chunk_id']}" for c in chunks]
    documents = [c["content"] for c in chunks]
    metadatas = [c["metadata"] for c in chunks]
    
    # เพิ่มชื่อไฟล์ลงใน Metadata ของทุก Chunk
    for m in metadatas:
        m["file_name"] = Path(pdf_path).name

    vector_store.add_documents(ids=ids, documents=documents, metadatas=metadatas)
    
    print("--- ✅ Ingestion Complete! Ready for RAG ---")

if __name__ == "__main__":
    # รองรับการโยนไฟล์ผ่าน Command Line
    target_pdf = sys.argv[1] if len(sys.argv) > 1 else "TINS-B784CBRZ (CHANG).pdf"
    
    # กรณีรันใน Windows แล้ว Path มีปัญหา ให้ลอง Hardcode Path เทสตรงนี้ได้
    # target_pdf = r"C:\Users\ASUS\Desktop\TINS-B784CBRZ (CHANG).pdf"
    
    run_pipeline(target_pdf)