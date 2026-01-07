import chromadb
import os
import uuid
from typing import List, Optional, Dict, Any

# พยายาม import Document object เพื่อ return ค่ากลับไปให้ RAG
try:
    from langchain_core.documents import Document
except ImportError:
    # Fallback ถ้าไม่มี lib langchain
    class Document:
        def __init__(self, page_content, metadata):
            self.page_content = page_content
            self.metadata = metadata

class VectorStore:
    def __init__(self, persist_directory="chroma_db"):
        # ตั้งค่าให้ ChromaDB บันทึกข้อมูลลงโฟลเดอร์
        self.client = chromadb.PersistentClient(path=persist_directory)
        
        # 1. สร้าง Collection สำหรับ TEXT (เนื้อหาเอกสาร)
        self.text_collection = self.client.get_or_create_collection(
            name="document_chunks",
            metadata={"hnsw:space": "cosine"}
        )
        
        # 2. สร้าง Collection สำหรับ IMAGES (รูปภาพ)
        self.image_collection = self.client.get_or_create_collection(
            name="image_chunks",
            metadata={"hnsw:space": "cosine"} 
        )
        
        print(f"✅ VectorStore initialized at '{persist_directory}'")

    def add_documents(self, ids: List[str], documents: List[str], metadatas: List[Dict[str, Any]]):
        """
        รับข้อมูลเข้าสู่ ChromaDB
        ids: รายชื่อ ID ของแต่ละ chunk
        documents: เนื้อหาข้อความ (Markdown)
        metadatas: ข้อมูลกำกับ (เช่น heading, file_name, page)
        """
        try:
            self.collection.add(
                ids=ids,
                documents=documents,
                metadatas=metadatas
            )
            print(f"บันทึกข้อมูลสำเร็จ {len(ids)} รายการ")
        except Exception as e:
            print(f"เกิดข้อผิดพลาดในการบันทึกข้อมูล: {e}")

    def query(self, query_texts: List[str], n_results: int = 5):
        """ค้นหาข้อมูลจากฐานข้อมูล"""
        return self.collection.query(
            query_texts=query_texts,
            n_results=n_results
        )