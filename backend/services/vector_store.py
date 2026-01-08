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
        
        # [FIX] ใช้ชื่อ text_collection ให้ชัดเจน
        self.text_collection = self.client.get_or_create_collection(
            name="document_chunks",
            metadata={"hnsw:space": "cosine"}
        )
        self.image_collection = self.client.get_or_create_collection(
            name="image_chunks",
            metadata={"hnsw:space": "cosine"} 
        )
        print(f"✅ VectorStore initialized at '{persist_directory}'")

    def add_documents(self, ids: List[str], documents: List[str], metadatas: List[Dict[str, Any]]):
        try:
            # [FIX] แก้ self.collection -> self.text_collection
            self.text_collection.add(
                ids=ids,
                documents=documents,
                metadatas=metadatas
            )
            print(f"   Saved {len(ids)} text chunks to DB.")
        except Exception as e:
            print(f"   Error saving documents: {e}")

    def query_text(self, query_text: str, n_results: int = 5, where: Optional[Dict] = None):
        return self.text_collection.query(
            query_texts=[query_text],
            n_results=n_results,
            where=where
        )

# =========================================================
# GLOBAL HELPERS (เพิ่มให้ ingest_doc.py เรียกใช้ได้)
# =========================================================

_store_instance = VectorStore()

def index_chunks(chunks: List[Any]):
    """บันทึก Chunks ลง DB"""
    text_ids, text_docs, text_metas = [], [], []
    
    for chunk in chunks:
        # กรองเฉพาะ Text และ Table
        if getattr(chunk, "source", "text") in ["text", "table"]:
            c_id = getattr(chunk, "id", str(uuid.uuid4()))
            content = getattr(chunk, "content", "")
            
            # แปลง Metadata ให้ ChromaDB รับได้ (ห้ามมี dict ซ้อน)
            raw_meta = getattr(chunk, "metadata", {})
            clean_meta = {}
            for k, v in raw_meta.items():
                if isinstance(v, (str, int, float, bool)):
                    clean_meta[k] = v
                else:
                    clean_meta[k] = str(v)
            
            # เพิ่ม Field สำคัญสำหรับการค้นหา
            clean_meta["source"] = getattr(chunk, "source", "text")
            clean_meta["doc_id"] = getattr(chunk, "doc_id", "unknown")
            clean_meta["page"] = getattr(chunk, "page", 1) or 1
            
            text_ids.append(str(c_id))
            text_docs.append(content)
            text_metas.append(clean_meta)

    if text_ids:
        _store_instance.add_documents(text_ids, text_docs, text_metas)

def search_similar(query: str, k: int = 5, doc_ids: Optional[List[str]] = None, sources: Optional[List[str]] = None, doc_types: Optional[List[str]] = None) -> List[Document]:
    """ค้นหาข้อมูล (Adapter สำหรับ RAG)"""
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

    results = _store_instance.query_text(query, n_results=k, where=final_where)
    
    docs = []
    if results and results.get('documents') and len(results['documents']) > 0:
        docs_list = results['documents'][0]
        metas_list = results['metadatas'][0] if results.get('metadatas') else [{}] * len(docs_list)
        for content, meta in zip(docs_list, metas_list):
            docs.append(Document(page_content=content, metadata=meta))
    return docs