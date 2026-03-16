# backend/services/rag.py
import litellm
from typing import List, Optional, Dict, Any
from .vector_store import search_vector_db
from ingestion.config import CUSTOM_API_BASE, CUSTOM_API_KEY, CUSTOM_MODEL_NAME

async def answer_question(
    query: str,
    doc_ids: Optional[List[str]] = None,
    top_k: int = 5,
    mode: str = "auto",
    history: Optional[List[Dict[str, str]]] = None  # [FIXED] เพิ่มรองรับ history
) -> Dict[str, Any]:
    
    # 1. ค้นหาข้อมูลที่เกี่ยวข้องจาก Vector DB (ChromaDB)
    search_results = search_vector_db(query, doc_ids=doc_ids, top_k=top_k)
    
    context_text = "\n---\n".join([res["content"] for res in search_results])
    
    # 2. สร้าง Prompt สำหรับ Qwen
    system_prompt = f"""คุณคือผู้ช่วยอัจฉริยะ ตอบคำถามจากบริบทที่ให้ไว้เท่านั้น 
หากในบริบทไม่มีคำตอบ ให้บอกว่าไม่ทราบ ห้ามเดา
ใช้ภาษาไทยที่สุภาพและเป็นทางการ

บริบท (Context):
{context_text}
"""
    
    messages = [{"role": "system", "content": system_prompt}]
    
    # [FIXED] ถ้ามีประวัติการสนทนา ให้ใส่เข้าไปใน messages
    if history:
        messages.extend(history)
        
    messages.append({"role": "user", "content": query})

    try:
        # 3. เรียกใช้ Qwen 2.5 72B (Local)
        response = litellm.completion(
            model=f"openai/{CUSTOM_MODEL_NAME}",
            messages=messages,
            api_base=CUSTOM_API_BASE,
            api_key=CUSTOM_API_KEY,
            temperature=0.2
        )
        
        answer = response.choices[0].message.content
        
        return {
            "answer": answer,
            "sources": search_results,
            "intent": "rag_query",
            "mode": mode
        }
    except Exception as e:
        print(f"⚠️ [RAG-Error] AI Failed: {e}")
        return {
            "answer": f"ขออภัย เกิดข้อผิดพลาดในการเชื่อมต่อกับ AI: {str(e)}",
            "sources": [],
            "intent": "error",
            "mode": mode
        }