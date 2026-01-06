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

    def add_documents(self, ids, documents, metadatas):
        if not ids: return
        
        # Upsert logic (ลบก่อนเขียน)
        existing = self.text_collection.get(ids=ids)
        if existing and existing["ids"]:
            self.text_collection.delete(ids=existing["ids"])
            
        self.text_collection.add(ids=ids, documents=documents, metadatas=metadatas)
        print(f"   Saved {len(ids)} text chunks.")

    def add_images(self, ids, embeddings, metadatas):
        if not ids: return

        existing = self.image_collection.get(ids=ids)
        if existing and existing["ids"]:
            self.image_collection.delete(ids=existing["ids"])
            
        self.image_collection.add(ids=ids, embeddings=embeddings, metadatas=metadatas)
        print(f"   Saved {len(ids)} image chunks.")

    def query_text(self, query_text, n_results=5, where=None):
        return self.text_collection.query(
            query_texts=[query_text],
            n_results=n_results,
            where=where
        )

# =========================================================
# GLOBAL INSTANCE & HELPER FUNCTIONS (เพื่อแก้ ImportError)
# =========================================================

# 1. สร้าง Instance กลางไว้ใช้ร่วมกัน
_store_instance = VectorStore()

# 2. ฟังก์ชัน search_similar (ที่ rag.py เรียกหา)
def search_similar(
    query: str,
    k: int = 5,
    doc_ids: Optional[List[str]] = None,
    sources: Optional[List[str]] = None,
    doc_types: Optional[List[str]] = None
) -> List[Document]:
    """
    ค้นหาข้อมูลที่เกี่ยวข้อง (Wrapper function)
    สร้าง where clause จาก parameters แล้วเรียก _store_instance
    """
    
    # สร้าง Filter (ChromaDB Where Clause)
    where_conditions = []
    
    if doc_ids:
        if len(doc_ids) == 1:
            where_conditions.append({"doc_id": doc_ids[0]})
        else:
            where_conditions.append({"doc_id": {"$in": doc_ids}})
            
    if sources:
        # กรองเฉพาะ source ที่ระบุ (เช่น text, table)
        if len(sources) == 1:
            where_conditions.append({"source": sources[0]})
        else:
            where_conditions.append({"source": {"$in": sources}})

    # รวม Filter
    final_where = None
    if len(where_conditions) == 1:
        final_where = where_conditions[0]
    elif len(where_conditions) > 1:
        final_where = {"$and": where_conditions}
    
    # Query จาก Text Collection (หลัก)
    results = _store_instance.query_text(query_text=query, n_results=k, where=final_where)
    
    # แปลงผลลัพธ์เป็น Document object list
    docs = []
    if results and results['documents']:
        for i, content in enumerate(results['documents'][0]):
            meta = results['metadatas'][0][i] if results['metadatas'] else {}
            docs.append(Document(page_content=content, metadata=meta))
            
    return docs

# 3. ฟังก์ชัน index_chunks (ที่ ingest_doc.py เรียกหา)
def index_chunks(chunks: List[Any]):
    """
    แยกประเภท Chunk (Text vs Image) แล้วบันทึกลง Collection ที่ถูกต้อง
    """
    text_ids, text_docs, text_metas = [], [], []
    image_ids, image_embeds, image_metas = [], [], []
    
    for chunk in chunks:
        # สร้าง ID ถ้าไม่มี
        c_id = chunk.metadata.get("chunk_id") or str(uuid.uuid4())
        
        # ตรวจสอบว่าเป็น Image Chunk หรือไม่ (ดูจาก metadata หรือ embedding)
        # (สมมติว่า Image Chunk จะมี 'image_embedding' field หรือ source='image')
        is_image = chunk.metadata.get("source") == "image" and "image_embedding" in chunk.metadata
        
        if is_image:
            # กรณีเป็นรูป (มี Vector มาแล้ว)
            embed = chunk.metadata.pop("image_embedding", None) # ดึง Embedding ออกจาก meta
            if embed:
                image_ids.append(c_id)
                image_embeds.append(embed)
                image_metas.append(chunk.metadata)
        else:
            # กรณีเป็น Text / Table
            text_ids.append(c_id)
            text_docs.append(chunk.page_content)
            text_metas.append(chunk.metadata)
            
    # บันทึก
    if text_ids:
        _store_instance.add_documents(text_ids, text_docs, text_metas)
    if image_ids:
        _store_instance.add_images(image_ids, image_embeds, image_metas)