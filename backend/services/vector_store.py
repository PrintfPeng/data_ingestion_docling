import chromadb
from chromadb.config import Settings
import os

class VectorStore:
    def __init__(self, persist_directory="chroma_db"):
        # ตั้งค่าให้ ChromaDB บันทึกข้อมูลลงโฟลเดอร์
        self.client = chromadb.PersistentClient(path=persist_directory)
        
        # 1. สร้าง Collection สำหรับ TEXT (เนื้อหาเอกสาร)
        # ใช้ HNSW (Graph-based) เพื่อการค้นหาที่รวดเร็ว
        self.text_collection = self.client.get_or_create_collection(
            name="document_chunks",
            metadata={"hnsw:space": "cosine"} # ใช้ Cosine Similarity สำหรับ Text
        )
        
        # 2. สร้าง Collection สำหรับ IMAGES (รูปภาพ) 🖼️
        # แยกออกมาเพราะ Embedding Space ของรูปกับ Text (OpenCLIP vs BGE-M3) ไม่เหมือนกัน
        self.image_collection = self.client.get_or_create_collection(
            name="image_chunks",
            metadata={"hnsw:space": "cosine"} 
        )
        
        print(f"✅ VectorStore initialized at '{persist_directory}'")
        print(f"   - Text Collection: {self.text_collection.count()} docs")
        print(f"   - Image Collection: {self.image_collection.count()} images")

    def add_documents(self, ids, documents, metadatas):
        """
        บันทึก Text ลงใน Text Collection
        """
        if not ids:
            return
            
        # ลบข้อมูลเก่าที่ id ซ้ำกันออกก่อน (Upsert)
        existing_ids = self.text_collection.get(ids=ids)["ids"]
        if existing_ids:
            self.text_collection.delete(ids=existing_ids)
            
        self.text_collection.add(
            ids=ids,
            documents=documents, # ChromaDB จะทำ Embedding ให้เองถ้าเราไม่ส่ง embeddings ไป
            metadatas=metadatas
        )
        print(f"บันทึกข้อมูล Text สำเร็จ {len(ids)} รายการ")

    def add_images(self, ids, embeddings, metadatas):
        """
        บันทึก Image Vector ลงใน Image Collection 🖼️
        """
        if not ids:
            return

        # ลบข้อมูลเก่าออกก่อน
        existing_ids = self.image_collection.get(ids=ids)["ids"]
        if existing_ids:
            self.image_collection.delete(ids=existing_ids)
            
        # บันทึกลง Collection
        self.image_collection.add(
            ids=ids,
            embeddings=embeddings, # เราส่ง Vector ที่คำนวณจาก CLIP ไปตรงๆ
            metadatas=metadatas
        )
        print(f"บันทึกข้อมูล Image สำเร็จ {len(ids)} รายการ")

    def query_similar_documents(self, query_text, n_results=5):
        """
        ค้นหา Text ที่ใกล้เคียง
        """
        results = self.text_collection.query(
            query_texts=[query_text],
            n_results=n_results
        )
        return results

    def query_similar_images(self, query_embedding, n_results=3):
        """
        ค้นหารูปภาพที่ใกล้เคียง (รับ Vector เข้ามา)
        """
        results = self.image_collection.query(
            query_embeddings=[query_embedding], # ส่ง Vector เข้าไปค้น
            n_results=n_results
        )
        return results