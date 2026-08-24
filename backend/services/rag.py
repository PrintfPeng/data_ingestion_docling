# backend/services/rag.py
import litellm
from typing import List, Optional, Dict, Any
from .vector_store import search_similar
from ingestion.config import CUSTOM_API_BASE, CUSTOM_API_KEY, CUSTOM_MODEL_NAME


def _docs_to_sources(docs) -> List[Dict[str, Any]]:
    """แปลง langchain Document เป็น dict ที่ frontend/main.py ใช้งานต่อได้"""
    sources = []
    for d in docs:
        md = dict(d.metadata or {})
        sources.append({
            "content": d.page_content,
            "doc_id": md.get("doc_id"),
            "page": md.get("page"),
            "source": md.get("source", "text"),
            "metadata": md,
        })
    return sources


async def answer_question(
    query: str,
    doc_ids: Optional[List[str]] = None,
    top_k: int = 5,
    mode: str = "auto",
    history: Optional[List[Dict[str, str]]] = None,  # รองรับ chat history
) -> Dict[str, Any]:

    # 1. ค้นหาข้อมูลที่เกี่ยวข้องจาก Vector DB (ChromaDB)
    docs = search_similar(query=query, k=top_k, doc_ids=doc_ids)
    search_results = _docs_to_sources(docs)

    context_text = "\n---\n".join([res["content"] for res in search_results])

    # 2. สร้าง Prompt สำหรับ Qwen
    system_prompt = f"""คุณคือผู้ช่วยอัจฉริยะ ตอบคำถามจากบริบทที่ให้ไว้เท่านั้น
หากในบริบทไม่มีคำตอบ ให้บอกว่าไม่ทราบ ห้ามเดา
ใช้ภาษาไทยที่สุภาพและเป็นทางการ

บริบท (Context):
{context_text}
"""

    messages = [{"role": "system", "content": system_prompt}]

    # ถ้ามีประวัติการสนทนา ให้ใส่เข้าไปใน messages
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
            temperature=0.2,
        )

        answer = response.choices[0].message.content

        return {
            "answer": answer,
            "sources": search_results,
            "intent": "rag_query",
            "mode": mode,
        }
    except Exception as e:
        print(f"⚠️ [RAG-Error] AI Failed: {e}")
        return {
            "answer": f"ขออภัย เกิดข้อผิดพลาดในการเชื่อมต่อกับ AI: {str(e)}",
            "sources": [],
            "intent": "error",
            "mode": mode,
        }
