from __future__ import annotations

import os
import re
import json
import logging
import asyncio
from typing import Dict, List, Optional, Tuple
from pathlib import Path
from difflib import SequenceMatcher

from dotenv import load_dotenv

# LangChain / Gemini chat wrapper used in project
# keep import local inside functions so file still loads if lib missing
try:
    from langchain_google_genai import ChatGoogleGenerativeAI
    from langchain_core.messages import HumanMessage, SystemMessage
    _HAS_GENAI = True
except Exception:
    ChatGoogleGenerativeAI = None  # type: ignore
    HumanMessage = None  # type: ignore
    SystemMessage = None  # type: ignore
    _HAS_GENAI = False

# --- [NEW] Re-ranking imports ---
try:
    from sentence_transformers import CrossEncoder
    _HAS_RERANKER = True
    _RERANK_MODEL = None  # Lazy load
except ImportError:
    _HAS_RERANKER = False
    _RERANK_MODEL = None

from .vector_store import search_similar

logger = logging.getLogger(__name__)

# -------------------------------------------------------------------
# paths & env
# -------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[2]
INGESTED_DIR = PROJECT_ROOT / "ingested"

_QNA_CACHE: Dict[str, List[Dict[str, str]]] = {}

# load .env to make sure key available (but do not raise here)
load_dotenv(override=True)
_GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

# LLM defaults
_LL_MODEL_FAST = "gemini-2.5-flash"
_LL_MODEL_SMALL = "gemini-2.0-mini"  # fallback if you want to set a smaller model in future
_DEFAULT_TEMPERATURE = 0.2

# Re-ranking config
_RERANK_MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"  # Fast & Accurate

# Q&A detection/extraction regex (flexible)
_QNA_PATTERN = re.compile(
    r"(?:\d+\s*[\.\-\)]\s*)?"                      # optional numbered prefix "1.", "1)"
    r"(?:ถาม|q|question)\s*[:\-]?\s*"              # question marker (Thai/eng)
    r"(?P<q>.+?)\s*"
    r"(?:ตอบ|a|answer)\s*[:\-]?\s*"
    r"(?P<a>.+?)(?=(?:\d+\s*[\.\-\)]\s*)?(?:ถาม|q|question)\s*[:\-]?|\Z)",
    re.IGNORECASE | re.DOTALL,
)


# -------------------------------------------------------------------
# Helper: LLM (Gemini) safe getter
# -------------------------------------------------------------------
def _get_llm_instance(model: Optional[str] = None, temperature: float = _DEFAULT_TEMPERATURE):
    """
    คืน ChatGoogleGenerativeAI instance หรือ None (ถ้าไม่มี API KEY หรือ lib)
    ไม่ raise เพื่อให้ระบบยังทำงานแบบ deterministic ได้แม้ไม่มี LLM
    """
    if not _HAS_GENAI:
        logger.debug("[rag] langchain_google_genai not installed -> no LLM available")
        return None
    api_key = os.getenv("GOOGLE_API_KEY") or _GOOGLE_API_KEY
    if not api_key:
        logger.debug("[rag] GOOGLE_API_KEY not set -> no LLM available")
        return None

    model = model or _LL_MODEL_FAST
    try:
        return ChatGoogleGenerativeAI(
            model=model,
            temperature=temperature,
            google_api_key=api_key,
        )
    except Exception as e:
        logger.exception("[rag] Failed to init LLM: %s", e)
        return None


# -------------------------------------------------------------------
# Helper: Reranker Model (Lazy Load)
# -------------------------------------------------------------------
def _get_reranker_model():
    """Load CrossEncoder model once"""
    global _RERANK_MODEL
    if not _HAS_RERANKER:
        return None
    
    if _RERANK_MODEL is None:
        try:
            logger.info(f"[rag] Loading Re-ranking model: {_RERANK_MODEL_NAME}")
            _RERANK_MODEL = CrossEncoder(_RERANK_MODEL_NAME, max_length=512)
        except Exception as e:
            logger.error(f"[rag] Failed to load reranker: {e}")
            return None
    return _RERANK_MODEL


# -------------------------------------------------------------------
# 1) Intent: rule-based fast path
# -------------------------------------------------------------------
def _rule_based_intent(query: str) -> Optional[str]:
    if not query or not query.strip():
        return None
    q = query.lower()

    table_keywords = ["ตาราง", "table", "คอลัมน์", "column", "แถว", "row", "สรุป", "summary", "ยอด", "amount"]
    image_keywords = ["รูป", "รูปภาพ", "image", "logo", "กราฟ", "graph", "chart", "diagram"]

    is_table = any(w in q for w in table_keywords)
    is_image = any(w in q for w in image_keywords)

    if is_table and not is_image:
        return "table"
    if is_image and not is_table:
        return "both"
    if is_table and is_image:
        return "both"
    return "text"


# -------------------------------------------------------------------
# 2) LLM-based intent classification (async)
# -------------------------------------------------------------------
async def classify_query_intent(query: str) -> str:
    """
    Try to use LLM to decide intent; fallback to rule-based.
    Keep this short to reduce cost.
    """
    # [UPDATED] เพื่อประหยัด Quota ช่วงนี้ ให้ใช้ Rule-based ไปเลย 100%
    # จะได้ไม่เสีย request ฟรีๆ ไปกับการเดา intent
    return _rule_based_intent(query) or "text"


# -------------------------------------------------------------------
# 3) Build context text
# -------------------------------------------------------------------
def _build_context_text(docs) -> str:
    parts: List[str] = []
    for i, d in enumerate(docs, 1):
        md = d.metadata or {}
        doc_id = md.get("doc_id", "unknown")
        page = md.get("page", "?")
        src = md.get("source", "text")
        doc_type = md.get("doc_type") or "unknown"
        header = f"[{i}] doc_id={doc_id} page={page} source={src} doc_type={doc_type}"
        content = getattr(d, "page_content", "") or getattr(d, "content", "") or ""
        # cut each chunk to avoid too long
        parts.append(f"{header}\n{content[:3000]}")
    joined = "\n\n".join(parts)
    return joined[:12000]


# -------------------------------------------------------------------
# Helper: Generate Fallback Snippets
# -------------------------------------------------------------------
def _generate_fallback_answer(docs, error_msg: str = "") -> str:
    """
    สร้างคำตอบสำรองจากเอกสารดิบ เมื่อ AI พังหรือ Quota เต็ม
    """
    if not docs:
        return "ไม่พบข้อมูลที่เกี่ยวข้องในเอกสาร (และ AI ไม่สามารถประมวลผลได้ในขณะนี้)"

    snippets = []
    for i, d in enumerate(docs[:4], 1): # เอาแค่ 4 อันแรกพอจะได้ไม่รก
        content = getattr(d, "page_content", "") or getattr(d, "content", "") or ""
        md = d.metadata or {}
        page = md.get('page', '?')
        # จัด Format ให้อ่านง่าย
        snippet_text = content[:400].replace("\n", " ").strip() + "..."
        snippets.append(f"**{i}. (หน้า {page})** {snippet_text}")
    
    joined_snippets = "\n\n".join(snippets)
    
    header = "⚠️ **แจ้งเตือน:** ขณะนี้โควต้า AI เต็มหรือระบบขัดข้อง " \
             "ระบบจึงดึงเนื้อหาที่เกี่ยวข้องจากเอกสารมาแสดงให้โดยตรงครับ:\n\n"
             
    return header + joined_snippets


# -------------------------------------------------------------------
# [NEW] Advanced Re-ranking Logic
# -------------------------------------------------------------------
def _rerank_documents(query: str, docs: list, top_k: int) -> list:
    """
    1. Exact Match Boosting: ให้คะแนนพิเศษถ้ามีคำตรงเป๊ะ (รหัส/ปี)
    2. Cross-Encoder Re-ranking: เรียงลำดับตามความเข้าใจภาษา (ถ้ามี model)
    """
    if not docs:
        return []

    # 1. Exact Match Boosting (Lite Hybrid)
    # ช่วยกรณีรหัสนักศึกษา/ตัวเลข ที่ Vector อาจมองว่าไม่สำคัญ
    query_terms = set(query.lower().split())
    
    # คำนวณคะแนนเบื้องต้น (Keyword overlap)
    scored_docs = []
    for d in docs:
        content = (getattr(d, "page_content", "") or "").lower()
        keyword_score = 0
        for term in query_terms:
            if term in content:
                keyword_score += 1.5  # ให้คะแนนพิเศษคำละ 1.5 แต้ม
        
        # เก็บ doc ไว้คู่กับคะแนน (Keyword score อย่างเดียวก่อน)
        scored_docs.append({"doc": d, "score": keyword_score})

    # 2. Cross-Encoder Re-ranking (The Real Intelligence)
    reranker = _get_reranker_model()
    if reranker:
        try:
            # เตรียมคู่ประโยค (Query, Doc) ส่งให้ AI ตัดสิน
            pairs = [[query, d["doc"].page_content] for d in scored_docs]
            ai_scores = reranker.predict(pairs)
            
            # รวมคะแนน AI + Keyword
            for i, score in enumerate(ai_scores):
                scored_docs[i]["score"] += float(score)
                
            logger.debug(f"[rag] Re-ranking complete for {len(docs)} docs.")
        except Exception as e:
            logger.warning(f"[rag] Re-ranking failed (using basic sort): {e}")

    # 3. Sort & Cut
    # เรียงจากคะแนนมากไปน้อย
    scored_docs.sort(key=lambda x: x["score"], reverse=True)
    
    # คืนค่าเฉพาะ Document object
    final_docs = [item["doc"] for item in scored_docs]
    
    return final_docs[:top_k]


# -------------------------------------------------------------------
# 4) Q&A extraction + matching utilities
# -------------------------------------------------------------------
def _load_qna_pairs_for_doc(doc_id: str) -> List[Dict[str, str]]:
    """
    Load ingested text.json and extract Q/A pairs with caching.
    """
    if doc_id in _QNA_CACHE:
        return _QNA_CACHE[doc_id]

    path = INGESTED_DIR / doc_id / "text.json"
    if not path.exists():
        _QNA_CACHE[doc_id] = []
        return []

    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        _QNA_CACHE[doc_id] = []
        return []

    # combine all text blocks
    full = "\n".join((item.get("content") or "") for item in raw)
    pairs: List[Dict[str, str]] = []
    for m in _QNA_PATTERN.finditer(full):
        q = " ".join(m.group("q").split())
        a = " ".join(m.group("a").split())
        if q and a:
            pairs.append({"question": q, "answer": a})
    _QNA_CACHE[doc_id] = pairs
    return pairs


def _simple_similarity(a: str, b: str) -> float:
    return SequenceMatcher(None, a, b).ratio()


def _find_best_qna_answer_from_docs(query: str, docs) -> Optional[Dict]:
    """
    For any docs that are qna-type (doc_type == 'qna'), try deterministic match:
    return {'answer': str, 'sources': [...] } or None
    """
    qna_doc_ids = sorted({
        (d.metadata or {}).get("doc_id")
        for d in docs
        if ((d.metadata or {}).get("doc_type") or "").lower() == "qna"
    })
    qna_doc_ids = [d for d in qna_doc_ids if d]
    if not qna_doc_ids:
        return None

    best_score = 0.0
    best_answer = None
    best_doc = None

    for doc_id in qna_doc_ids:
        pairs = _load_qna_pairs_for_doc(doc_id)
        for p in pairs:
            score = _simple_similarity(query, p["question"])
            if score > best_score:
                best_score = score
                best_answer = p["answer"]
                best_doc = doc_id

    # [UPDATED] Threshold lowered to 0.45 for more flexibility (user typos/partial match)
    if not best_answer or best_score < 0.45:
        return None

    # build sources list from docs that match doc_id
    sources = []
    for d in docs:
        md = d.metadata or {}
        if md.get("doc_id") == best_doc:
            sources.append({
                "doc_id": md.get("doc_id"),
                "page": md.get("page"),
                "source": md.get("source"),
                "chunk_id": md.get("chunk_id"),
            })
    return {"answer": best_answer, "sources": sources, "score": best_score}


# -------------------------------------------------------------------
# 5) main RAG function
# -------------------------------------------------------------------
async def answer_question(
    query: str,
    doc_ids: Optional[List[str]] = None,
    top_k: int = 10,
    mode: str = "auto",  # auto | text | table | both
) -> Dict:
    """
    Robust RAG flow:
    - intent decision (rule-based -> optional LLM)
    - similarity search + RE-RANKING (Improved Logic)
    - deterministic Q&A (if doc_type qna)
    - LLM RAG fallback (if needed) with safe error handling
    """

    if not query or not query.strip():
        return {"answer": "คำถามว่าง กรุณาพิมพ์ข้อความคำถาม", "sources": [], "intent": None, "mode": mode}

    # 1) intent
    if mode == "auto":
        intent = _rule_based_intent(query) or "text"
        # [UPDATED] ปิด LLM classify เพื่อประหยัด Quota (ใช้ rule-based ด้านบน)
    elif mode in ("text", "table", "both"):
        intent = mode
    else:
        intent = _rule_based_intent(query) or "text"

    # map to sources
    if intent == "text":
        sources_filter = ["text"]
    elif intent == "table":
        sources_filter = ["table", "text"]
    else:
        sources_filter = ["text", "table"]

    # Decide doc_types to prefer: if user provided doc_ids, try those; else None
    doc_types = None

    # 2) Search + Re-ranking Strategy
    try:
        # Step A: Fetch MORE candidates (x3 of needed) to ensure we don't miss anything
        # (เพราะ Vector Search อาจเอาของดีไปไว้อันดับ 15)
        initial_k = top_k * 3
        
        raw_docs = search_similar(
            query=query,
            k=initial_k, 
            doc_ids=doc_ids,
            sources=sources_filter,
            doc_types=doc_types,
        )
        
        # Step B: Re-rank using Cross-Encoder + Keyword Boosting
        # คัดเนื้อๆ เน้นๆ เหลือแค่ top_k ตามที่ขอ
        docs = _rerank_documents(query, raw_docs, top_k)
        
    except Exception as e:
        logger.exception("[rag] search_similar failed: %s", e)
        return {
            "answer": "เกิดข้อผิดพลาดระหว่างค้นหาใน Vector DB. ลองล้าง index แล้ว re-ingest หรือดู log เพิ่มเติม.",
            "sources": [],
            "intent": intent,
            "mode": mode,
        }

    if not docs:
        return {
            "answer": "ไม่พบข้อมูลที่เกี่ยวข้องเพียงพอในฐานข้อมูลเอกสาร",
            "sources": [],
            "intent": intent,
            "mode": mode,
        }

    # 2.5 deterministic Q&A on qna docs
    qna_direct = _find_best_qna_answer_from_docs(query, docs)
    if qna_direct:
        return {
            "answer": qna_direct["answer"],
            "sources": qna_direct["sources"],
            "intent": intent,
            "mode": f"{mode}+qna_direct",
        }

    # 3) Prepare context and choose qna_mode for LLM prompt
    context_text = _build_context_text(docs)
    
    # --- [UPDATED LOGIC] ---
    # เชื่อ Metadata (doc_type) ที่ Auto-detect มาจากขั้นตอน Ingestion
    qna_mode = False
    
    qna_doc_count = 0
    for d in docs:
        md = d.metadata or {}
        # ตรวจสอบ doc_type จาก metadata (case-insensitive)
        if (md.get("doc_type") or "").lower() == "qna":
            qna_doc_count += 1
            
    # ถ้าเอกสารที่ค้นเจอเป็น QnA เกิน 40% ให้ถือว่าเป็นโหมด QnA
    if docs and (qna_doc_count / len(docs)) > 0.4:
        qna_mode = True
    # -----------------------

    # Build system prompt
    if qna_mode:
        system_prompt = (
            "คุณกำลังตอบคำถามจากเอกสารที่เป็นชุด 'ถาม: ... / ตอบ: ...' (ภาษาไทย/อังกฤษ)\n"
            "กติกา:\n"
            "1) ให้หาคู่ 'ถาม: ... / ตอบ: ...' ที่ตรงกับคำถามของผู้ใช้มากที่สุด\n"
            "2) ใช้ข้อความหลัง 'ตอบ:' เป็นคำตอบหลัก (สามารถปรับรูป but not change meaning)\n"
            "3) ห้ามใช้ความรู้ภายนอกเอกสาร\n"
            "4) ถ้าไม่มีคำตอบที่เกี่ยวข้อง ให้ตอบว่า 'ไม่พบในเอกสาร'\n\n"
            "=== CONTEXT START ===\n"
            f"{context_text}\n"
            "=== CONTEXT END ===\n"
        )
    else:
        system_prompt = (
            "คุณเป็นผู้ช่วยอ่านและตอบคำถามจาก CONTEXT ด้านล่าง (ภาษาไทย/อังกฤษ). "
            "ให้ตอบโดยอ้างอิงเฉพาะข้อมูลใน CONTEXT เท่านั้น และถ้าข้อมูลไม่พอ ให้ตอบว่า 'ไม่ทราบจากข้อมูลที่มีอยู่'.\n\n"
            f"(intent={intent})\n"
            "=== CONTEXT START ===\n"
            f"{context_text}\n"
            "=== CONTEXT END ===\n"
        )
    user_prompt = query

    # 4) If no LLM available -> give safe fallback answer using context snippets
    llm = _get_llm_instance()
    if not llm:
        # Fallback: AI not configured
        ans = _generate_fallback_answer(docs, "No LLM Configured")
        
        # prepare sources
        sources = [{
            "doc_id": (d.metadata or {}).get("doc_id"),
            "page": (d.metadata or {}).get("page"),
            "source": (d.metadata or {}).get("source"),
            "chunk_id": (d.metadata or {}).get("chunk_id"),
        } for d in docs[:top_k]]
        return {
            "answer": ans,
            "sources": sources,
            "intent": intent,
            "mode": f"{mode}+no_llm",
        }

    # 5) Use LLM, but guard errors and do a single retry for transient errors
    answer_text = None
    llm_attempts = 0
    max_attempts = 2
    last_exc = None
    
    while llm_attempts < max_attempts:
        try:
            resp = await llm.ainvoke([SystemMessage(content=system_prompt), HumanMessage(content=user_prompt)])
            answer_text = getattr(resp, "content", None) or str(resp)
            break
        except Exception as e:
            llm_attempts += 1
            last_exc = e
            logger.warning("[rag] LLM call failed (attempt %d/%d): %s", llm_attempts, max_attempts, e)
            
            # [UPDATED] ถ้าเจอ Quota เต็ม หรือ Rate Limit ให้ Fallback ไปโชว์เนื้อหาดิบเลย ไม่ต้องรอ Retry
            msg = str(e).lower()
            if "quota" in msg or "rate limit" in msg or "exceeded" in msg or "429" in msg:
                logger.warning("[rag] Quota limit hit! Switching to fallback snippets.")
                answer_text = _generate_fallback_answer(docs, "Quota Exceeded")
                break
            
            # small backoff before retrying transient errors
            await asyncio.sleep(1 + llm_attempts * 1.0)

    if answer_text is None:
        # [UPDATED] สุดท้ายถ้าพังจริงๆ ก็ยังโชว์ fallback snippets แทนที่จะบอกว่า error
        logger.exception("[rag] LLM failed finally: %s", last_exc)
        answer_text = _generate_fallback_answer(docs, "AI Error")

    # 6) prepare sources for frontend
    sources = []
    for d in docs:
        md = d.metadata or {}
        sources.append({
            "doc_id": md.get("doc_id"),
            "page": md.get("page"),
            "source": md.get("source"),
            "chunk_id": md.get("chunk_id"),
        })

    return {
        "answer": answer_text,
        "sources": sources,
        "intent": intent,
        "mode": f"{mode}+qna_llm" if qna_mode else f"{mode}+rag_llm",
    }