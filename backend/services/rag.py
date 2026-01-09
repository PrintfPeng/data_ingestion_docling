import google.generativeai as genai
import time
import random
from ingestion.config import GOOGLE_API_KEY
from .vector_store import search_similar

def generate_with_retry(model, prompt, max_retries=3):
    """
    ฟังก์ชันสำหรับยิง Gemini แบบมีการรอถ้าเจอ Error 429
    """
    for attempt in range(max_retries):
        try:
            response = model.generate_content(prompt)
            return response.text
        except Exception as e:
            error_msg = str(e)
            # เช็คว่าเป็น Error 429 (Quota) หรือ 503 (Server Busy) ไหม
            if "429" in error_msg or "429" in str(e) or "Quota exceeded" in error_msg:
                wait_time = (2 ** attempt) + random.uniform(0, 1) # Exponential Backoff: 1s, 2s, 4s...
                print(f"[RAG] Quota exceeded. Retrying in {wait_time:.2f}s... (Attempt {attempt+1}/{max_retries})")
                time.sleep(wait_time)
            else:
                # ถ้าเป็น Error อื่น (เช่น Key ผิด) ให้โยน Error ออกไปเลย ไม่ต้องรอ
                raise e
    
    return "ขออภัยครับ ระบบ AI กำลังทำงานหนักเกินขีดจำกัด กรุณารอสักครู่แล้วลองใหม่ (Error 429)"

def answer_question(question: str) -> dict:
    """
    ค้นหาข้อมูลและตอบคำถาม (พร้อมระบบ Retry)
    """
    # 1. Search Vector DB
    try:
        relevant_docs = search_similar(question, k=5)
    except Exception as e:
        print(f"[RAG] Search error: {e}")
        return {
            "answer": f"เกิดข้อผิดพลาดในการค้นหาฐานข้อมูล: {e}",
            "sources": [],
            "intent": "error"
        }
    
    # 2. Prepare Context
    context_text = ""
    sources_data = []
    
    for doc in relevant_docs:
        # ดึง content และ metadata
        content = doc.page_content.replace("\n", " ")
        meta = doc.metadata if hasattr(doc, "metadata") else {}
        
        source_type = meta.get("source", "unknown")
        page = meta.get("page", "?")
        doc_id = meta.get("doc_id", "?")
        
        # เก็บ Source ส่งกลับ Frontend
        source_info = {
            "content": content[:300] + "...", # ตัดให้สั้นลงหน่อย
            "source": source_type,
            "page": page,
            "doc_id": doc_id,
            "metadata": meta # ส่งไปทั้งก้อน เผื่อ Frontend ใช้ file_path
        }
        sources_data.append(source_info)
        
        # Context สำหรับ AI
        context_text += f"- [{source_type.upper()}] (Page {page}): {content}\n"

    # 3. Generate Answer
    if not GOOGLE_API_KEY:
        return {
            "answer": "ไม่พบ API Key (GOOGLE_API_KEY) ในการตั้งค่า", 
            "sources": sources_data,
            "intent": "config_error"
        }
        
    try:
        genai.configure(api_key=GOOGLE_API_KEY)
        model = genai.GenerativeModel("gemini-2.0-flash")
        
        prompt = (
            f"Context information is below.\n"
            f"---------------------\n"
            f"{context_text}\n"
            f"---------------------\n"
            f"Given the context information and not prior knowledge, answer the query.\n"
            f"If the context contains descriptions of images or tables relevant to the answer, please mention them explicitly.\n"
            f"Query: {question}\n"
            f"Answer (in Thai):"
        )
        
        # เรียกใช้ผ่านฟังก์ชัน Retry ที่เขียนไว้ข้างบน
        answer = generate_with_retry(model, prompt)
        
    except Exception as e:
        answer = f"เกิดข้อผิดพลาดในการสร้างคำตอบ: {str(e)}"

    # 4. Return
    return {
        "answer": answer,
        "sources": sources_data,
        "intent": "qa"
    }