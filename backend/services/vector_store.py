import os
import chromadb
from typing import List, Optional, Dict, Any

class VectorStore:
    def __init__(self, collection_name: str = "docling_collection"):
        # กำหนดที่เก็บฐานข้อมูล ChromaDB
        self.db_path = os.path.join(os.getcwd(), "chroma_db")
        self.client = chromadb.PersistentClient(path=self.db_path)
        
        # สร้างหรือดึง Collection เดิมมาใช้
        self.collection = self.client.get_or_create_collection(name=collection_name)

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