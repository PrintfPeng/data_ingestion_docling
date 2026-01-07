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
        
        # =======================================================
# ส่วนที่เพิ่ม: Adapter เพื่อให้ใช้งานร่วมกับ rag.py ได้
# =======================================================

# 1. สร้างคลาส Document หลอกๆ (เพื่อให้ rag.py ใช้งานได้)
class Document:
    def __init__(self, page_content, metadata):
        self.page_content = page_content
        self.metadata = metadata

# 2. สร้างตัวแปร Global ไว้ใช้
_store_instance = VectorStore()

# 3. สร้างฟังก์ชัน search_similar ที่ rag.py ตามหา
def search_similar(query: str, k: int = 5, **kwargs):
    """
    ฟังก์ชันแปลงการเรียกใช้จาก rag.py ให้เข้ากับ VectorStore ตัวเก่า
    """
    # เรียกใช้ query แบบเดิม
    results = _store_instance.query(query_texts=[query], n_results=k)
    
    # แปลงผลลัพธ์จาก Dictionary เป็น List of Documents
    documents = []
    
    # เช็คว่ามีข้อมูลไหม (ChromaDB จะคืนค่าเป็น list ซ้อน list)
    if results and results.get('documents') and len(results['documents']) > 0:
        docs_list = results['documents'][0]
        metas_list = results['metadatas'][0] if results.get('metadatas') else [{}] * len(docs_list)
        
        for content, meta in zip(docs_list, metas_list):
            documents.append(Document(page_content=content, metadata=meta))
            
    return documents