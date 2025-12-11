from __future__ import annotations

from typing import Dict, List, Optional
import os
import re
from difflib import SequenceMatcher

from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, SystemMessage

from .vector_store import search_similar


# -------------------------------------------------------------------
# ตั้งค่า LLM (Gemini)
# -------------------------------------------------------------------

# โหลด .env ให้ทับค่าเดิม (กัน Ghost Key)
load_dotenv(override=True)
_GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

if not _GOOGLE_API_KEY:
    raise RuntimeError(
        "GOOGLE_API_KEY is not set. Please add it to your .env or environment."
    )


def _get_llm() -> ChatGoogleGenerativeAI:
    """
    เตรียม LLM (Gemini) สำหรับตอบคำถาม
    """
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",  # <--- ใส่ชื่อโมเดลตรงๆ ที่นี่
        temperature=0.2,           # <--- ใส่ค่า temperature ตรงๆ ที่นี่
        google_api_key=_GOOGLE_API_KEY,
    )
    return llm


# -------------------------------------------------------------------
# 1) Rule-based intent (ถูก ๆ เร็ว ๆ ก่อน)
# -------------------------------------------------------------------
def _rule_based_intent(query: str) -> Optional[str]:
    """
    เดา intent แบบ rule-based ง่าย ๆ จาก keyword
    คืนค่า: "text" | "table" | "both" | None
    """
    q = query.lower().strip()
    if not q:
        return None

    table_keywords = [
        "ตาราง", "table", "สรุปข้อมูล", "สรุปผล", "สถิติ", 
        "สรุปคะแนน", "แถวที่", "คอลัมน์", "column", "row", 
        "ชีท", "sheet",
    ]

    image_keywords = [
        "รูป", "รูปภาพ", "image", "logo", "โลโก้", 
        "กราฟ", "graph", "chart", "แผนภาพ", "diagram", "แผนภูมิ",
    ]

    is_table = any(kw in q for kw in table_keywords)
    is_image = any(kw in q for kw in image_keywords)

    if is_table and not is_image:
        return "table"
    if is_image and not is_table:
        return "both"
    if is_table and is_image:
        return "both"

    return "text"


# -------------------------------------------------------------------
# 2) LLM-based intent (ละเอียดแต่แพงกว่า)
# -------------------------------------------------------------------
async def classify_query_intent(query: str) -> str:
    """
    ใช้ LLM ช่วยบอกว่า query ต้องไปดู text / table / both
    """
    llm = _get_llm()

    system_prompt = (
        "คุณเป็นตัวจัดประเภทคำถามเกี่ยวกับเอกสาร PDF หลายประเภท\n"
        "เป้าหมายคือบอกว่าเมื่อจะตอบคำถามนี้ เราควรโฟกัสข้อมูลจากไหนเป็นหลัก:\n"
        "- text  = เนื้อหาบรรยาย / ย่อหน้า / ข้อความยาว ๆ\n"
        "- table = ข้อมูลในตาราง เช่น แถว-คอลัมน์ รายการ สรุปตัวเลข\n"
        "- both  = ต้องใช้ทั้งข้อความและข้อมูลตารางร่วมกัน\n\n"
        "ให้ตอบสั้น ๆ เป็นคำเดียวเท่านั้น หนึ่งใน: text, table, both.\n"
    )

    user_prompt = f"คำถาม: {query}\n\nตอบแค่หนึ่งคำ: text, table หรือ both"

    try:
        resp = await llm.ainvoke(
            [
                SystemMessage(content=system_prompt),
                HumanMessage(content=user_prompt),
            ]
        )
        raw = (resp.content or "").strip().lower()
    except Exception:
        return "text"

    if "both" in raw:
        return "both"
    if "table" in raw:
        return "table"
    return "text"


# -------------------------------------------------------------------
# 3) รวม context จากเอกสาร
# -------------------------------------------------------------------
def _build_context_text(docs) -> str:
    parts: List[str] = []
    for i, d in enumerate(docs, start=1):
        meta = d.metadata or {}
        doc_id = meta.get("doc_id", "unknown")
        page = meta.get("page", "?")
        source = meta.get("source", "text")
        doc_type = meta.get("doc_type") or "unknown"
        header = (
            f"[{i}] (doc_id={doc_id}, page={page}, "
            f"source={source}, doc_type={doc_type})"
        )
        parts.append(f"{header}\n{d.page_content}")

    joined = "\n\n".join(parts)
    return joined[:12000]


def _is_qna_like(docs) -> bool:
    if not docs:
        return False
    text_all = "".join(d.page_content or "" for d in docs)
    text_all = text_all.replace(" ", "")
    return ("ถาม:" in text_all) and ("ตอบ:" in text_all)


# -------------------------------------------------------------------
# 3.1 Q&A helper
# -------------------------------------------------------------------
_QNA_PATTERN = re.compile(
    r"ถาม:\s*(.+?)\s*ตอบ:\s*(.+?)(?=\n\s*ถาม:|$)",
    re.DOTALL,
)

def _extract_qna_pairs(text: str) -> List[Dict[str, str]]:
    text_norm = text.replace("\r\n", "\n")
    pairs: List[Dict[str, str]] = []
    for m in _QNA_PATTERN.finditer(text_norm):
        q = m.group(1).strip()
        a = m.group(2).strip()
        if q and a:
            pairs.append({"question": q, "answer": a})
    return pairs


def _find_best_qna_answer(query: str, context_text: str) -> Optional[str]:
    pairs = _extract_qna_pairs(context_text)
    if not pairs:
        return None

    best_score = 0.0
    best_answer: Optional[str] = None

    for p in pairs:
        q_text = p["question"]
        score = SequenceMatcher(None, query, q_text).ratio()
        if score > best_score:
            best_score = score
            best_answer = p["answer"]

    if best_score >= 0.55 and best_answer:
        return best_answer

    return None


# -------------------------------------------------------------------
# 4) main RAG function
# -------------------------------------------------------------------
async def answer_question(
    query: str,
    doc_ids: Optional[List[str]] = None,
    top_k: int = 10,
    mode: str = "auto",
) -> Dict:
    # 1) intent
    if mode == "auto":
        intent = _rule_based_intent(query)
        if intent is None:
            intent = await classify_query_intent(query)
    elif mode in ("text", "table", "both"):
        intent = mode
    else:
        intent = _rule_based_intent(query) or await classify_query_intent(query)

    if intent == "text":
        source_filter = ["text"]
    elif intent == "table":
        source_filter = ["table", "text"]
    elif intent == "both":
        source_filter = ["text", "table"]
    else:
        source_filter = None

    # 2) search
    docs = search_similar(
        query=query,
        k=top_k,
        doc_ids=doc_ids,
        sources=source_filter,
    )

    if not docs:
        return {
            "answer": "ไม่พบข้อมูลที่เกี่ยวข้องเพียงพอในฐานข้อมูลเอกสาร",
            "sources": [],
            "intent": intent,
            "mode": mode,
        }

    # 3) context & prompt
    context_text = _build_context_text(docs)
    qna_mode = _is_qna_like(docs)

    if qna_mode:
        direct_ans = _find_best_qna_answer(query, context_text)
        if direct_ans:
            sources = []
            for d in docs:
                meta = d.metadata or {}
                sources.append(
                    {
                        "doc_id": meta.get("doc_id"),
                        "page": meta.get("page"),
                        "source": meta.get("source"),
                        "chunk_id": meta.get("chunk_id"),
                    }
                )
            return {
                "answer": direct_ans,
                "sources": sources,
                "intent": intent,
                "mode": f"{mode}+qna_direct",
            }

    if qna_mode:
        system_prompt = (
            "คุณกำลังตอบคำถามจากเอกสารที่เป็น 'ชุดคำถาม-คำตอบ' ภาษาไทย\n"
            "กติกา:\n"
            "1) ให้หาคู่ 'ถาม: ... / ตอบ: ...' ที่สอดคล้องกับคำถามของผู้ใช้มากที่สุด\n"
            "2) ให้ตอบโดยใช้ข้อความหลังคำว่า 'ตอบ:' เป็นคำตอบหลัก\n"
            "3) ห้ามใช้ความรู้จากภายนอก ห้ามเดา\n"
            "4) ถ้าไม่พบคำตอบให้ตอบว่า 'ไม่พบในเอกสาร'\n\n"
            "=== CONTEXT (Q&A) START ===\n"
            f"{context_text}\n"
            "=== CONTEXT (Q&A) END ===\n"
        )
    else:
        system_prompt = (
            "คุณเป็นผู้ช่วยอ่านและวิเคราะห์เอกสาร PDF\n"
            "ให้ตอบคำถามโดยอ้างอิงเฉพาะจาก CONTEXT ด้านล่างนี้เท่านั้น\n"
            "ห้ามใช้ความรู้จากภายนอกเอกสาร และห้ามเดาเกินข้อมูลที่มี\n"
            "ถ้า CONTEXT ไม่มีข้อมูลเพียงพอจริง ๆ ให้ตอบว่า 'ไม่ทราบจากข้อมูลที่มีอยู่'\n\n"
            f"(query intent: {intent}, mode: {mode})\n\n"
            "=== CONTEXT START ===\n"
            f"{context_text}\n"
            "=== CONTEXT END ===\n\n"
            "ตอบคำถามให้กระชับ ชัดเจน และอ้างอิงจากเนื้อหาใน CONTEXT เท่านั้น"
        )

    user_prompt = query

    llm = _get_llm()
    resp = await llm.ainvoke(
        [
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_prompt),
        ]
    )
    answer_text = resp.content if hasattr(resp, "content") else str(resp)

    sources = []
    for d in docs:
        meta = d.metadata or {}
        sources.append(
            {
                "doc_id": meta.get("doc_id"),
                "page": meta.get("page"),
                "source": meta.get("source"),
                "chunk_id": meta.get("chunk_id"),
            }
        )

    return {
        "answer": answer_text,
        "sources": sources,
        "intent": intent,
        "mode": mode if not qna_mode else f"{mode}+qna_llm",
    }