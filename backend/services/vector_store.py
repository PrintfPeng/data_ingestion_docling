import chromadb
import os
import uuid
from typing import List, Optional, Dict, Any

# Adapter สำหรับ LangChain document
try:
    from langchain_core.documents import Document
except ImportError:
    class Document:
        def __init__(self, page_content, metadata):
            self.page_content = page_content
            self.metadata = metadata

class VectorStore:
    def __init__(self, persist_directory="chroma_db"):
        self.client = chromadb.PersistentClient(path=persist_directory)
        
        # Collection หลัก (เก็บ Text, Table, และ Image Description รวมกัน)
        self.text_collection = self.client.get_or_create_collection(
            name="document_chunks",
            metadata={"hnsw:space": "cosine"}
        )
        # Collection สำรอง (เผื่อใช้อนาคต)
        self.image_collection = self.client.get_or_create_collection(
            name="image_chunks",
            metadata={"hnsw:space": "cosine"} 
        )

    def add_documents(self, ids: List[str], documents: List[str], metadatas: List[Dict[str, Any]]):
        try:
            self.text_collection.add(
                ids=ids,
                documents=documents,
                metadatas=metadatas
            )
            print(f"   Saved {len(ids)} chunks to DB.")
        except Exception as e:
            print(f"   Error saving documents: {e}")

    def query_text(self, query_text: str, n_results: int = 5, where: Optional[Dict] = None):
        return self.text_collection.query(
            query_texts=[query_text],
            n_results=n_results,
            where=where
        )

# =========================================================
# GLOBAL HELPERS
# =========================================================

def index_chunks(chunks: List[Any]):
    """บันทึก Chunks ลง DB"""
    store = VectorStore()
    
    text_ids, text_docs, text_metas = [], [], []
    
    for chunk in chunks:
        # [FIX] เพิ่ม "image" เข้าไปใน list ที่ยอมรับ
        if getattr(chunk, "source", "text") in ["text", "table", "image"]:
            c_id = getattr(chunk, "id", str(uuid.uuid4()))
            content = getattr(chunk, "content", "")
            
            # ดึง Metadata
            raw_meta = getattr(chunk, "metadata", {})
            clean_meta = {}
            for k, v in raw_meta.items():
                # แปลงค่าให้เป็น string/int/float/bool เท่านั้น
                if isinstance(v, (str, int, float, bool)):
                    clean_meta[k] = v
                else:
                    clean_meta[k] = str(v)
            
            # เพิ่ม Field สำคัญ
            clean_meta["source"] = getattr(chunk, "source", "text")
            clean_meta["doc_id"] = getattr(chunk, "doc_id", "unknown")
            clean_meta["page"] = getattr(chunk, "page", 1) or 1
            
            # [NEW] สำหรับ Image ให้แน่ใจว่ามี file_path
            if getattr(chunk, "source", "") == "image":
                clean_meta["file_path"] = getattr(chunk, "file_path", "")

            text_ids.append(str(c_id))
            text_docs.append(content)
            text_metas.append(clean_meta)

    if text_ids:
        store.add_documents(text_ids, text_docs, text_metas)

def search_similar(
    query: str, 
    k: int = 5, 
    doc_ids: Optional[List[str]] = None, 
    sources: Optional[List[str]] = None, 
    doc_types: Optional[List[str]] = None
) -> List[Document]:
    """ค้นหาข้อมูล (Adapter สำหรับ RAG)"""
    store = VectorStore()
    
    where_conditions = []
    if doc_ids:
        if len(doc_ids) == 1: where_conditions.append({"doc_id": doc_ids[0]})
        else: where_conditions.append({"doc_id": {"$in": doc_ids}})
    
    if sources:
        if len(sources) == 1: where_conditions.append({"source": sources[0]})
        else: where_conditions.append({"source": {"$in": sources}})
        
    final_where = None
    if len(where_conditions) == 1: final_where = where_conditions[0]
    elif len(where_conditions) > 1: final_where = {"$and": where_conditions}

    results = store.query_text(query, n_results=k, where=final_where)
    
    docs = []
    if results and results.get('documents') and len(results['documents']) > 0:
        docs_list = results['documents'][0]
        metas_list = results['metadatas'][0] if results.get('metadatas') else [{}] * len(docs_list)
        for content, meta in zip(docs_list, metas_list):
            docs.append(Document(page_content=content, metadata=meta))
    return docs