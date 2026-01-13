# backend/services/rag.py
from google import genai 
import re
import os
from pathlib import Path
from ingestion.config import GOOGLE_API_KEY
from .vector_store import search_similar

# Configuration
MODEL_NAME = "gemini-2.5-flash"
BASE_DIR = Path(r"D:\DATA_INGES")
INGESTED_PATH = BASE_DIR / "ingested"

# Initialize Client (New Library Style)
client = None
if GOOGLE_API_KEY:
    try:
        client = genai.Client(api_key=GOOGLE_API_KEY)
    except Exception as e:
        print(f"[RAG] ⚠️ Init Error: {e}")

def answer_question(question: str) -> dict:
    print(f"\n[RAG] 🔍 Query: {question}")
    
    # ---------------------------------------------------------
    # 1. Search Vector DB
    # ---------------------------------------------------------
    try:
        relevant_docs = search_similar(question, k=5)
    except Exception as e:
        print(f"[RAG] ❌ DB Error: {e}")
        return {"answer": "เกิดข้อผิดพลาดในการค้นหาข้อมูล", "sources": [], "intent": "error"}
    
    context_text = ""
    sources_data = []
    doc_pages_map = {} 
    
    for doc in relevant_docs:
        meta = doc.metadata if hasattr(doc, "metadata") else {}
        content = doc.page_content.replace("\n", " ")
        doc_id = meta.get("doc_id")
        
        try: page = int(meta.get("page", 0))
        except: page = 0
            
        sources_data.append({
            "content": content[:200] + "...", 
            "page": page,
            "doc_id": doc_id,
            "metadata": meta 
        })
        context_text += f"Document: {doc_id} (Page {page})\nContent: {content}\n\n"

        if doc_id:
            if doc_id not in doc_pages_map: doc_pages_map[doc_id] = set()
            doc_pages_map[doc_id].add(page)

    # ---------------------------------------------------------
    # 2. Scan Directory for Images
    # ---------------------------------------------------------
    related_images = []
    processed_urls = set()

    if INGESTED_PATH.exists():
        for doc_id, pages in doc_pages_map.items():
            images_dir = INGESTED_PATH / doc_id / "images"
            if not images_dir.exists(): continue

            for img_file in images_dir.iterdir():
                if img_file.suffix.lower() not in ['.png', '.jpg', '.jpeg']: continue
                match = re.search(r'(?:_p|page_?)(\d+)', img_file.name)
                if match:
                    try:
                        file_page = int(match.group(1))
                        if file_page in pages:
                            img_url = f"/ingested/{doc_id}/images/{img_file.name}"
                            if img_url not in processed_urls:
                                processed_urls.add(img_url)
                                related_images.append({
                                    "url": img_url,
                                    "page": file_page,
                                    "doc_id": doc_id
                                })
                                print(f"[RAG] 📸 Found image: {img_file.name}")
                    except: continue

    # ---------------------------------------------------------
    # 3. Generate Answer (New Library)
    # ---------------------------------------------------------
    answer = "ไม่สามารถเชื่อมต่อกับ AI ได้ (API Key Error)"
    
    if client:
        try:
            prompt = (
                f"You are a helpful assistant. Use the following Context to answer the Question in Thai.\n"
                f"Context:\n{context_text}\n\n"
                f"Question: {question}\n"
                f"Answer:"
            )

            # Call API แบบใหม่
            response = client.models.generate_content(
                model=MODEL_NAME,
                contents=prompt
            )
            
            # ดึง text ออกมา
            if response.text:
                answer = response.text
            else:
                answer = "AI ไม่ตอบกลับ (No text generated)"

        except Exception as e:
            error_msg = str(e)
            print(f"[RAG] ❌ AI Error: {error_msg}")
            answer = f"Error generating answer: {error_msg}"

    return {
        "answer": answer,
        "sources": sources_data,
        "related_images": related_images, 
        "intent": "qa"
    }