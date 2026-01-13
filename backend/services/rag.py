# backend/services/rag.py
import google.generativeai as genai
import re
import os
from pathlib import Path
from ingestion.config import GOOGLE_API_KEY
from .vector_store import search_similar

# [จุดสำคัญที่ 2] ตรวจสอบ path นี้ให้ตรงกับเครื่องคุณ
INGESTED_PATH = Path(r"D:\DATA_INGES\ingested")

def answer_question(question: str) -> dict:
    print(f"\n[RAG] 🔍 Query: {question}")
    
    # 1. Search Vector DB
    try:
        relevant_docs = search_similar(question, k=5)
    except Exception as e:
        return {"answer": f"Error searching database: {e}", "sources": [], "intent": "error"}
    
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
        context_text += f"--- Page {page} ---\n{content}\n\n"

        if doc_id and page > 0:
            if doc_id not in doc_pages_map: doc_pages_map[doc_id] = set()
            doc_pages_map[doc_id].add(page)

    # 2. Scan Directory for Images (หารูปจากโฟลเดอร์)
    related_images = []
    processed_urls = set()

    for doc_id, pages in doc_pages_map.items():
        images_dir = INGESTED_PATH / doc_id / "images"
        if images_dir.exists():
            for img_file in images_dir.iterdir():
                if img_file.suffix.lower() not in ['.png', '.jpg', '.jpeg']: continue
                
                # Regex หาเลขหน้า
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
                    except: pass

    # 3. Generate Answer (แก้ Error 404 ด้วยการลองหลายๆ Model)
    answer = "ไม่สามารถเชื่อมต่อกับ AI ได้ในขณะนี้"
    
    if GOOGLE_API_KEY:
        try:
            genai.configure(api_key=GOOGLE_API_KEY)
            
            # [จุดสำคัญที่ 3] รายชื่อโมเดลสำรอง (ถ้าตัวแรกไม่ได้ จะลองตัวถัดไป)
            models_to_try = [
                "gemini-2.0-flash",           # ใหม่ล่าสุด
                "gemini-2.0-flash-lite-preview-02-05", 
                "gemini-1.5-flash",           # มาตรฐาน
                "gemini-1.5-flash-001",       # ชื่อเต็ม (บางทีต้องใช้ตัวนี้)
                "gemini-1.5-flash-latest"     
            ]
            
            success = False
            last_error = ""

            for model_name in models_to_try:
                try:
                    # print(f"[RAG] Trying model: {model_name}...")
                    model = genai.GenerativeModel(model_name)
                    response = model.generate_content(
                        f"Context:\n{context_text}\n\nQuestion: {question}\nAnswer (in Thai):"
                    )
                    answer = response.text
                    success = True
                    break # หยุดถ้าสำเร็จ
                except Exception as e:
                    last_error = str(e)
                    continue
            
            if not success:
                print(f"[RAG] All models failed. Last error: {last_error}")
                answer = f"ขออภัย เกิดข้อผิดพลาดจาก Google AI (404/Quota): {last_error}"

        except Exception as e:
            answer = f"System Error: {str(e)}"

    return {
        "answer": answer,
        "sources": sources_data,
        "related_images": related_images, 
        "intent": "qa"
    }