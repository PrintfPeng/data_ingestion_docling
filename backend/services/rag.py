from __future__ import annotations

import os
import re
import json
import logging
import math
import asyncio
from typing import Dict, List, Optional, Any
from pathlib import Path
from difflib import SequenceMatcher

from dotenv import load_dotenv

# [CHANGE] เปลี่ยน Import เป็น ChatOpenAI สำหรับ Custom API
try:
    from langchain_openai import ChatOpenAI
    # [UPDATED] เพิ่ม AIMessage เพื่อรองรับ History
    from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
    _HAS_GENAI = True
except Exception:
    ChatOpenAI = None  # type: ignore
    HumanMessage = None  # type: ignore
    SystemMessage = None  # type: ignore
    AIMessage = None # type: ignore
    _HAS_GENAI = False

# --- Re-ranking imports ---
try:
    from sentence_transformers import CrossEncoder
    _HAS_RERANKER = True
    _RERANK_MODEL = None  # Lazy load
except ImportError:
    _HAS_RERANKER = False
    _RERANK_MODEL = None

from .vector_store import search_similar, sanitize_doc_id

try:
    from langchain_google_genai import ChatGoogleGenerativeAI
except ImportError:
    ChatGoogleGenerativeAI = None

logger = logging.getLogger(__name__)

# -------------------------------------------------------------------
# paths & env
# -------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[2]
INGESTED_DIR = PROJECT_ROOT / "ingested"

_QNA_CACHE: Dict[str, List[Dict[str, str]]] = {}
_QNA_CACHE_MAX_SIZE = 100

# load .env to make sure key available
load_dotenv(override=True)


_CUSTOM_API_KEY = os.getenv("CUSTOM_API_KEY")
_CUSTOM_API_BASE = os.getenv("CUSTOM_API_BASE")

# [CHANGE] กำหนด Model เป็น Qwen ตามที่ต้องการ
_LL_MODEL_FAST = os.getenv("CUSTOM_MODEL_NAME", "qwen/qwen-2.5-72b-instruct")
# ใช้โมเดลเดียวกันเป็น Fallback หรือจะเปลี่ยนเป็นตัวเล็กกว่าถ้ามี
_LL_MODEL_SMALL = _LL_MODEL_FAST 

_DEFAULT_TEMPERATURE = 0.1 # ลด Temperature ลงเพื่อลดการมั่ว

# Re-ranking config
_RERANK_MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"

# [CONFIG] Thresholds ที่เข้มงวดขึ้น
MIN_SCORE_THRESHOLD = 0.25 # ต้องมีความมั่นใจระดับหนึ่ง
MIN_KEYWORD_OVERLAP = 1    # ต้องมีคำซ้ำอย่างน้อย 1 คำ (สำหรับคำถามยาว)

INTENT_THRESHOLDS = {
    "qna_match": 0.20,
    "table": 0.15,
    "text": 0.10,
    "both": 0.15
}

# Q&A detection regex
_QNA_PATTERN = re.compile(
    r"(?:\d+\s*[\.\-\)]\s*)?"
    r"(?:ถาม|q|question)\s*[:\-]?\s*"
    r"(?P<q>.+?)\s*"
    r"(?:ตอบ|a|answer)\s*[:\-]?\s*"
    r"(?P<a>.+?)(?=(?:\d+\s*[\.\-\)]\s*)?(?:ถาม|q|question)\s*[:\-]?|\Z)",
    re.IGNORECASE | re.DOTALL,
)

# Normalize Score Function
def normalize_score(raw_score: float) -> float:
    try:
        return 1 / (1 + math.exp(-raw_score))
    except OverflowError:
        return 0.0 if raw_score < 0 else 1.0

# -------------------------------------------------------------------
# Helper: Sanitization
# -------------------------------------------------------------------
def _sanitize_html_content(html: str) -> str:
    if not html: return ""
    html = re.sub(r"<script.*?>.*?</script>", "", html, flags=re.IGNORECASE | re.DOTALL)
    html = re.sub(r" on\w+=", " data-blocked-event=", html, flags=re.IGNORECASE)
    html = re.sub(r"javascript:", "blocked:", html, flags=re.IGNORECASE)
    return html


# -------------------------------------------------------------------
# Helper: LLM (Custom/OpenAI Compatible) safe getter
# -------------------------------------------------------------------
def _get_llm_instance(model: Optional[str] = None, temperature: float = _DEFAULT_TEMPERATURE):
    if not _HAS_GENAI:
        logger.debug("[rag] langchain_openai not installed -> no LLM available")
        return None
    
    # [CHANGE] รับค่า Key และ Base URL จาก Env ใหม่
    api_key = os.getenv("CUSTOM_API_KEY") or _CUSTOM_API_KEY
    api_base = os.getenv("CUSTOM_API_BASE") or _CUSTOM_API_BASE
    
    if not api_key:
        logger.debug("[rag] CUSTOM_API_KEY not set -> no LLM available")
        return None

    model = model or _LL_MODEL_FAST
    try:
        # [CHANGE] สร้าง ChatOpenAI Instance
        return ChatOpenAI(
            model=model,
            temperature=temperature,
            openai_api_key=api_key,
            openai_api_base=api_base,
            max_retries=2,
            request_timeout=60 # เพิ่ม timeout เผื่อโมเดลใหญ่ตอบช้า
            # max_tokens=150 # เพิ่มเพื่อรองรับคำตอบยาวขึ้น
        )
    except Exception as e:
        logger.exception("[rag] Failed to init LLM: %s", e)
        return None


# backend/services/rag.py

def _get_google_llm():
    """สร้าง Google Gemini Instance เป็นแผนสำรอง"""
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key or not ChatGoogleGenerativeAI:
        return None
    
    try:
        return ChatGoogleGenerativeAI(
            model="gemini-2.5-flash", # หรือ gemini-1.5-flash
            google_api_key=api_key,
            temperature=0.3,
            max_tokens=2048,
            convert_system_message_to_human=True # บางที Gemini ไม่ชอบ System msg
        )
    except Exception as e:
        logger.error(f"[rag] Failed to init Google LLM: {e}")
        return None

# -------------------------------------------------------------------
# Helper: Reranker Model (Lazy Load)
# -------------------------------------------------------------------
def _get_reranker_model():
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
# Intent & Logic Guards
# -------------------------------------------------------------------

def _detect_general_intent(query: str) -> bool:
    """ตรวจสอบว่าเป็นคำถามทั่วไปที่ไม่เกี่ยวกับเอกสารหรือไม่"""
    q = query.lower().strip()
    general_keywords = ["สวัสดี", "hello", "hi", "วันนี้วันอะไร", "อากาศ", "who are you", "คุณคือใคร", "สบายดีไหม"]
    if q in general_keywords:
        return True
    # เช็คคำถามเรื่องเวลา/วันที่แบบเจาะจง
    if "วันนี้" in q and "วันอะไร" in q:
        return True
    return False

def _keyword_overlap_count(query: str, text: str) -> int:
    """นับจำนวนคำที่ตรงกันระหว่าง Query และ Chunk Content (Simple Guardrail)"""
    q_clean = re.sub(r'[^\w\s]', '', query).lower()
    t_clean = re.sub(r'[^\w\s]', '', text).lower()
    
    q_tokens = set(q_clean.split())
    t_tokens = set(t_clean.split())
    
    stopwords = {"คือ", "เป็น", "อยู่", "จะ", "ได้", "ที่", "ซึ่ง", "อัน", "ของ", "what", "is", "are", "the", "a", "an", "ครับ", "ค่ะ"}
    q_tokens = q_tokens - stopwords
    
    if not q_tokens: return 0
    return len(q_tokens.intersection(t_tokens))

def _filter_relevant_docs(query: str, docs: list, min_score: float = MIN_SCORE_THRESHOLD) -> list:
    """
    กรองเอกสารที่ไม่เกี่ยวข้องออกอย่างเข้มงวด
    """
    passed = []
    for d in docs:
        score = d.metadata.get("ai_score", 0.0)
        content = d.page_content or ""
        
        # Guard 1: Score Threshold
        if score < min_score:
            continue
            
        # Guard 2: Keyword Overlap (ถ้า query ยาวพอสมควร)
        if len(query) > 10: 
            overlap = _keyword_overlap_count(query, content)
            if overlap < MIN_KEYWORD_OVERLAP:
                # ถ้าไม่มี Keyword ตรงเลย แต่ Score สูงมาก (Semantic Match) อาจจะยอมให้ผ่าน
                if score < 0.75: 
                    continue

        passed.append(d)
    return passed


# -------------------------------------------------------------------
# 3) Build context text
# -------------------------------------------------------------------
def _build_context_text(docs) -> str:
    parts: List[str] = []
    total_tokens = 0
    MAX_TOKENS_ESTIMATE = 4000

    parts.append("⚠️ **แหล่งข้อมูลอ้างอิง:** (เรียงตามความเกี่ยวข้อง)\n")

    for i, d in enumerate(docs, 1):
        content = getattr(d, "page_content", "") or getattr(d, "content", "") or ""
        content = content.replace("\x00", "") 
        
        if len(content) + total_tokens > MAX_TOKENS_ESTIMATE:
            break

        md = d.metadata or {}
        doc_id = md.get("doc_id", "unknown")
        page = md.get("page", "?")
        score = md.get("ai_score", 0.0)
        
        header = (f"[SOURCE {i}] ID: {doc_id} | Page: {page} | Score: {score:.2f}")
        parts.append(f"{header}\n{content[:3000]}")
        total_tokens += len(content[:3000])

    joined = "\n\n".join(parts)
    return joined


# -------------------------------------------------------------------
# Helper: Generate Fallback Snippets
# -------------------------------------------------------------------
def _generate_fallback_answer(docs, error_msg: str = "") -> str:
    if not docs:
        return "ไม่พบข้อมูลที่เกี่ยวข้องในเอกสาร (และ AI ไม่สามารถประมวลผลได้ในขณะนี้)"

    snippets = []
    for i, d in enumerate(docs[:4], 1):
        content = getattr(d, "page_content", "") or getattr(d, "content", "") or ""
        md = d.metadata or {}
        page = md.get('page', '?')
        snippet_text = content[:400].replace("\n", " ").strip() + "..."
        snippets.append(f"**{i}. (หน้า {page})** {snippet_text}")
    
    joined_snippets = "\n\n".join(snippets)
    header = f"⚠️ **แจ้งเตือน ({error_msg}):** ระบบจึงดึงเนื้อหาที่เกี่ยวข้องจากเอกสารมาแสดงให้โดยตรงครับ:\n\n"
              
    return header + joined_snippets

# -------------------------------------------------------------------
# [NEW] 1. Metadata Pre-filtering (Inference)
# -------------------------------------------------------------------
def _infer_search_filters(query: str) -> Dict[str, Any]:
    """
    วิเคราะห์ Query เพื่อสร้าง Filter สำหรับ ChromaDB
    โดย Map กับ metadata ที่ chunking.py/semantic_enricher.py สร้างไว้
    """
    filters = {}
    q = query.lower()

    # Intent Mapping (ดูจาก chunking.py)
    if any(w in q for w in ["ราคา", "บาท", "งบ", "cost", "price", "budget", "เงิน", "จ่าย"]):
        filters["primary_intent"] = "financial"
    
    # Section Mapping (ดูจาก semantic_enricher.py)
    if "สัญญา" in q or "contract" in q or "agreement" in q:
        filters["section"] = "legal_section"
    elif "ประชุม" in q or "meeting" in q or "minute" in q:
        filters["section"] = "work_section"
    
    return filters

# -------------------------------------------------------------------
# [NEW] 2. Fact-Checking (Self-Correction)
# -------------------------------------------------------------------
async def _verify_answer_factuality(llm, question: str, answer: str, context: str) -> str:
    """
    ให้ AI ตรวจสอบตัวเองว่าคำตอบที่ได้ มีหลักฐานใน Context จริงไหม
    """
    # ถ้าคำตอบสั้นมาก หรือบอกว่าไม่รู้ ไม่ต้องเช็ค
    if len(answer) < 50 or "ไม่พบข้อมูล" in answer:
        return answer

    verification_prompt = (
        "คุณคือผู้ตรวจสอบความถูกต้อง (Fact Checker) ที่เข้มงวด\n"
        "ภารกิจ: ตรวจสอบว่า 'คำตอบ' ด้านล่างนี้ มีหลักฐานสนับสนุนจาก 'CONTEXT' ที่ให้มาหรือไม่\n"
        "\n"
        f"--- CONTEXT ---\n{context[:4000]}...\n\n"
        f"--- คำถาม ---\n{question}\n\n"
        f"--- คำตอบที่ต้องตรวจสอบ ---\n{answer}\n\n"
        "กฎการตัดสิน:\n"
        "1. ถ้าคำตอบ **ถูกต้องและมีใน Context**: ให้ตอบกลับด้วยคำตอบเดิมเป๊ะๆ (ห้ามแก้)\n"
        "2. ถ้าคำตอบ **มีการมั่ว (Hallucination)** หรือข้อมูลไม่มีใน Context: ให้แก้คำตอบโดยใช้เฉพาะข้อมูลใน Context เท่านั้น\n"
        "3. ถ้าข้อมูลใน Context ไม่พอตอบ: ให้ตอบว่า 'ขออภัย ข้อมูลในเอกสารไม่เพียงพอต่อการตอบคำถามนี้'\n"
        "\n"
        "ผลลัพธ์ (คำตอบที่ผ่านการตรวจสอบแล้ว):"
    )

    try:
        res = await llm.ainvoke([HumanMessage(content=verification_prompt)])
        verified_answer = getattr(res, "content", str(res)).strip()
        
        # ป้องกันกรณี AI ตอบกลับมาสั้นเกินไปหรือ Error
        if len(verified_answer) < 5: 
            return answer
            
        logger.info("[RAG] ✅ Fact-Check completed.")
        return verified_answer
    except Exception as e:
        logger.warning(f"[RAG] Fact-Check failed: {e}")
        return answer # Fallback to original answer


# -------------------------------------------------------------------
# [UPDATED] Advanced Re-ranking Logic (Smarter)
# -------------------------------------------------------------------
def _clean_text_for_rerank(text: str) -> str:
    text = re.sub(r'\s+', ' ', text).strip()
    return text[:1000]

def _rerank_documents(query: str, docs: list, top_k: int) -> list:
    if not docs:
        return []

    # 1. Keyword Boosting
    query_terms = set(query.lower().split())
    scored_docs = []
    for d in docs:
        content = (getattr(d, "page_content", "") or "").lower()
        base_score = 0.0
        
        for term in query_terms:
            if term in content:
                base_score += 1.0
        
        if query.lower() in content:
            base_score += 5.0 # Boost หนักๆถ้าเจอ Exact phrase
            
        # Init metadata if missing
        if "ai_score" not in d.metadata:
            d.metadata["ai_score"] = 0.0
            
        d.metadata["keyword_score"] = base_score
        scored_docs.append(d)

    # 2. AI Re-ranking (Cross Encoder)
    reranker = _get_reranker_model()
    if reranker:
        try:
            valid_pairs_indices = []
            pairs = []
            
            for i, doc in enumerate(scored_docs):
                clean_content = _clean_text_for_rerank(doc.page_content)
                if clean_content:
                    pairs.append([query, clean_content])
                    valid_pairs_indices.append(i)
            
            if pairs:
                raw_scores = reranker.predict(pairs)
                
                for idx, raw in zip(valid_pairs_indices, raw_scores):
                    norm_score = normalize_score(float(raw))
                    
                    # Hybrid Score: AI (Major) + Keyword (Minor Boost)
                    final_score = norm_score
                    if scored_docs[idx].metadata["keyword_score"] > 2:
                         final_score += 0.1
                         
                    scored_docs[idx].metadata["ai_score"] = min(1.0, final_score)
                    scored_docs[idx].metadata["raw_score"] = float(raw)
                
                # Sort by AI Score
                scored_docs.sort(key=lambda x: x.metadata["ai_score"], reverse=True)
                return scored_docs[:top_k]

        except Exception as e:
            logger.warning(f"[rag] Re-ranking failed: {e}")

    # 3. Sort & Cut (Fallback to keyword score)
    scored_docs.sort(key=lambda x: x.metadata.get("keyword_score", 0), reverse=True)
    return scored_docs[:top_k]


# -------------------------------------------------------------------
# 4) Q&A extraction + matching utilities
# -------------------------------------------------------------------
def _load_qna_pairs_for_doc(doc_id: str) -> List[Dict[str, str]]:
    if len(_QNA_CACHE) > _QNA_CACHE_MAX_SIZE:
        _QNA_CACHE.clear()

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
    qna_doc_ids = sorted({
        (d.metadata or {}).get("doc_id")
        for d in docs
        if (d.metadata or {}).get("doc_id") 
    })
    qna_doc_ids = [d for d in qna_doc_ids if d]
    
    if not qna_doc_ids:
        return None

    all_pairs = []
    for doc_id in qna_doc_ids:
        pairs = _load_qna_pairs_for_doc(doc_id)
        for p in pairs:
            all_pairs.append({"question": p["question"], "answer": p["answer"], "doc_id": doc_id})

    if not all_pairs:
        return None

    best_score = 0.0
    best_item = None
    
    # Simple check
    for p in all_pairs:
        score = _simple_similarity(query, p["question"])
        if score > best_score:
            best_score = score
            best_item = p
        
    if best_item and best_score >= 0.75: # High confidence only
        return {
            "answer": best_item["answer"],
            "sources": [{"doc_id": best_item["doc_id"], "source": "Q&A Match", "page": "?"}],
            "score": float(best_score)
        }
    return None


# -------------------------------------------------------------------
# [NEW FUNCTION] Query Rewriting Logic
# -------------------------------------------------------------------
async def _rewrite_query_with_history(llm, query: str, history: List[Dict[str, str]]) -> str:
    """แปลงคำถาม Follow-up ให้เป็นคำถามเต็ม โดยดูจากประวัติ"""
    if not history:
        return query

    # แปลง history dict เป็น String
    history_context = ""
    # เอาแค่ 3 ข้อความล่าสุดก็พอ เพื่อไม่ให้ Prompt ยาวเกินไป
    for msg in history[-3:]: 
        role = "User" if msg.get("role") == "user" else "Assistant"
        content = msg.get("content", "").replace("\n", " ")
        history_context += f"{role}: {content}\n"

    prompt = (
        "ภารกิจ: แปลงคำถามของผู้ใช้ (Current Input) ให้เป็นประโยคคำถามที่สมบูรณ์และเข้าใจได้ด้วยตัวเอง "
        "โดยอ้างอิงบริบทจากการสนทนาก่อนหน้า (History) เพื่อนำไปใช้ค้นหาข้อมูลในระบบ\n"
        "ข้อควรระวัง: ห้ามตอบคำถาม! ให้ทำหน้าที่แค่เรียบเรียงประโยคใหม่เท่านั้น\n"
        "ถ้าคำถามสมบูรณ์อยู่แล้ว ให้ส่งคืนคำถามเดิมได้เลย\n\n"
        f"--- History ---\n{history_context}\n"
        f"--- Current Input ---\nUser: {query}\n"
        "--- Standalone Question (Thai) ---"
    )
    
    try:
        # ใช้ LLM ตัวเดียวกับที่ใช้ตอบคำถาม
        res = await llm.ainvoke([HumanMessage(content=prompt)])
        new_query = getattr(res, "content", str(res)).strip()
        logger.info(f"[RAG] 🔄 Rewritten Query: '{query}' -> '{new_query}'")
        return new_query
    except Exception as e:
        logger.error(f"[RAG] Rewrite failed: {e}")
        return query


# -------------------------------------------------------------------
# 5) main RAG function (UPGRADED & ROBUST & STATEFUL)
# -------------------------------------------------------------------
async def answer_question(
    query: str,
    doc_ids: Optional[List[str]] = None,
    top_k: int = 10,
    mode: str = "auto",
    history: List[Dict[str, str]] = [] # [NEW] รับ History เข้ามา
) -> Dict:
    
    # Init LLM เร็วขึ้นเพื่อใช้ Rewrite Query
    llm = _get_llm_instance(model=_LL_MODEL_FAST)

    # 1. Input Check
    if not query or not query.strip():
        return {"answer": "กรุณาพิมพ์คำถามครับ", "sources": [], "intent": None, "mode": mode}

    # [NEW] STEP 1: General Intent Guard
    if _detect_general_intent(query):
        return {
            "answer": "คำถามนี้ดูเหมือนเป็นคำถามทั่วไป ผมตอบได้เฉพาะข้อมูลที่มีในเอกสารที่แนบมาเท่านั้นครับ (ลองถามเกี่ยวกับเนื้อหาในเอกสารดูนะครับ)",
            "sources": [],
            "intent": "general",
            "mode": mode
        }

    # ---------------------------------------------------------
    # [NEW] STEP 1.5: Query Rewriting (หัวใจของ Stateful Search)
    # ---------------------------------------------------------
    search_query = query # ค่าตั้งต้น
    if history and llm:
        # ถ้ามีประวัติ ให้ AI ช่วยเกลาคำถามก่อนเอาไปค้น DB
        search_query = await _rewrite_query_with_history(llm, query, history)

    # [NEW] STEP 1.8: Infer Filters (Pre-filtering)
    filters = _infer_search_filters(search_query)
    if filters:
        logger.info(f"[RAG] 🔍 Applied Filters: {filters}")

    # [NEW] STEP 2: Mode Selection (Deterministic)
    if mode == "auto":
        q_lower = search_query.lower() # ใช้ search_query ที่เกลาแล้วในการตัดสินใจ
        if any(x in q_lower for x in ["ตาราง", "table", "ยอด", "สถิติ", "list", "รายการ", "สรุป"]):
            intent = "table"
        else:
            intent = "text"
    else:
        intent = mode

    # Sanitize doc_ids
    sanitized_doc_ids = None
    if doc_ids:
        sanitized_doc_ids = [sanitize_doc_id(doc_id) for doc_id in doc_ids if doc_id]

    # 3. Search (3-Layer Fallback Strategy)
    docs = []
    raw_docs = []

    try:
        # Layer 1: Strict Search
        # [IMPORTANT] ใช้ search_query (คำถามที่เกลาแล้ว) ในการค้นหา
        raw_docs = search_similar(
            search_query, 
            k=top_k*3, 
            doc_ids=sanitized_doc_ids, 
            where_filter=filters # ส่ง Filter ไป
        )
        
        logger.info(f"[rag] Found {len(raw_docs)} raw docs")

        # Q&A Check
        qna_match = _find_best_qna_answer_from_docs(search_query, raw_docs)
        if qna_match:
             return {
                 "answer": qna_match["answer"],
                 "sources": qna_match["sources"],
                 "intent": "qna",
                 "mode": f"{mode}+qna"
             }

        # Rerank (ใช้ search_query)
        docs = _rerank_documents(search_query, raw_docs, top_k)
        
        # [NEW] STEP 4: STRICT FILTERING (No Rescue Mission)
        # กรองอีกชั้นด้วย Python เพื่อความชัวร์
        filtered_docs = []
        for d in docs:
            score = d.metadata.get("ai_score", 0)
            threshold = MIN_SCORE_THRESHOLD
            
            # ถ้า intent ตรงกันเป๊ะ ยอมลด threshold ได้นิดหน่อย
            if filters and d.metadata.get("primary_intent") == filters.get("primary_intent"):
                threshold = 0.20 
            
            if score >= threshold:
                filtered_docs.append(d)
        
        if not filtered_docs:
            return {
                "answer": "ไม่พบข้อมูลที่ตรงกับคำถามในเอกสารที่แนบมาครับ (Relevance Score ต่ำเกินไป)",
                "sources": [],
                "intent": intent,
                "mode": mode
            }
            
        docs = filtered_docs # Use filtered docs

    except Exception as e:
        logger.error(f"[rag] Search failed: {e}")
        return {"answer": f"ระบบค้นหาขัดข้อง: {str(e)}", "sources": [], "intent": intent, "mode": mode}

# --- Prepare Context & Table Map (FIXED) ---
    table_map = {}
    table_cat_map = {}
    context_parts = [] # เก็บเนื้อหาทีละส่วนเพื่อรวมเป็น Context ใหญ่
    table_counter = 0 
    
    found_table_ids = []
    
    try:
        context_parts.append("⚠️ **แหล่งข้อมูลอ้างอิง:** (เรียงตามความเกี่ยวข้อง)\n")

        for i, d in enumerate(docs, 1):
            md = d.metadata or {}
            doc_id = md.get("doc_id", "unknown")
            page = md.get("page", "?")
            source = str(md.get("source", "text")).lower().strip()
            # ดึงเนื้อหา (Markdown) มาเตรียมไว้
            content = getattr(d, "page_content", "") or ""
            content = content.replace("\x00", "")
            
            # สร้างส่วนหัวของ Chunk นี้
            chunk_header = f"[SOURCE {i}] ID: {doc_id} | Page: {page}"

            if source == "table":
                table_counter += 1
                table_ref_id = str(table_counter) # เลขรัน 1, 2, 3...
                found_table_ids.append(table_ref_id)
                
                # 1. ดึง HTML มาเก็บใน Map
                raw_html = md.get("html_content", "")
                safe_html = _sanitize_html_content(raw_html)
                if not safe_html:
                    safe_html = f"<pre class='text-xs overflow-auto p-2 bg-gray-100'>{md.get('markdown_content', 'No content')}</pre>"
                
                # เก็บโดยใช้ TBL_i
                tbl_key = f"TBL_{table_ref_id}"
                table_map[tbl_key] = safe_html
                
                # 2. แปะป้ายบอก AI ชัดๆ
                chunk_header += f" | **TYPE: TABLE (Code: [SHOW_TABLE:{tbl_key}])**"
                
                # Mapping category/role
                category = md.get("category", "").strip().lower()
                role = md.get("role", "").strip().lower()
                if category:
                    cat_key = f"cat:{category}"
                    if cat_key not in table_cat_map: table_cat_map[cat_key] = safe_html
                if role:
                    role_key = f"role:{role}"
                    if role_key not in table_cat_map: table_cat_map[role_key] = safe_html
            
            # เพิ่มเนื้อหาลงใน Context Parts
            context_parts.append(f"{chunk_header}\n{content[:3500]}")

        # รวมทุกส่วนเป็นข้อความเดียวส่งให้ AI
        context_text = "\n\n".join(context_parts)

    except Exception as e:
        logger.error(f"[rag] Context build failed: {e}")
        return {"answer": "เกิดข้อผิดพลาดในการเตรียมข้อมูล", "sources": [], "intent": intent, "mode": mode}
    
    # ------------------------------------------------------------------
    # PROMPT ENGINEERING
    # ------------------------------------------------------------------
    
    if mode == "table":
        # === MODE 1: TABLE EXTRACTION ===
        system_prompt = (
            "บทบาท: คุณคือระบบ AI อัจฉริยะที่เชี่ยวชาญการสกัดข้อมูลโครงสร้าง (Structured Data Extraction)\n"
            "ภารกิจ: ค้นหา 'ตาราง' ที่ตรงกับคำถามของผู้ใช้มากที่สุดจาก CONTEXT ที่ให้มา\n"
            "\n"
            "ขั้นตอนการทำงาน:\n"
            "1. สแกนหาข้อมูลที่มีระบุว่าเป็น (SOURCE: Table) หรือ (Type: table)\n"
            "2. อ่านหัวข้อตาราง (Table Name/Summary) และเนื้อหาภายในเพื่อตรวจสอบความเกี่ยวข้อง\n"
            "3. การตอบกลับ (Strict Output):\n"
            "   - ถ้าเจอ: ให้ตอบเฉพาะรหัส [SHOW_TABLE:TBL_x] เท่านั้น (ห้ามพูด ห้ามเกริ่นนำ)\n"
            "   - ถ้าเจอหลายตารางที่เกี่ยวข้องกัน: ส่งมาให้ครบ เช่น [SHOW_TABLE:TBL_1] [SHOW_TABLE:TBL_2]\n"
            "   - ถ้าไม่เจอ: ให้ตอบว่า 'NULL'\n"
            "\n"
            f"=== CONTEXT ===\n{context_text}\n==============="
        )
    else:
        # === MODE 2: SMART ANALYST ===
        system_prompt = (
            "บทบาท: คุณคือ 'ผู้เชี่ยวชาญด้านเอกสาร' ที่เน้นความถูกต้องของข้อมูลสูงสุด\n"
            "หน้าที่: ตอบคำถามจาก Context ที่ให้มา โดยเลือกวิธีนำเสนอที่ดีที่สุด\n"
            "\n"
            "🧠 วิธีการเลือกแสดงผล (Decision Logic):\n"
            "1. **ถ้าเป็น 'ตารางข้อมูล' (Data Table):** เช่น รายรับรายจ่าย, สเปคสินค้า\n"
            "   - ให้ใช้ Tag: [SHOW_TABLE:TBL_x] (ห้ามวาดตาราง Markdown |...| เองเด็ดขาด!)\n"
            "2. **ถ้าเป็น 'แบบฟอร์ม' หรือ 'ตารางซับซ้อน':**\n"
            "   - ห้ามใช้ตาราง ให้ใช้รูปภาพแทน (ถ้ามี) หรือสรุปความเอา\n"
            "\n"
            "📋 รูปแบบการตอบ:\n"
            "1. ตอบคำถามให้ตรงประเด็น โดยพิจารณาจากทั้ง **History** และ **Context**\n"
            "2. แทรกหลักฐาน (Table) ตาม Logic ข้างบน\n"
            "3. อธิบายข้อมูลในหลักฐานนั้นสั้นๆ\n"
            "\n"
            "⚠️ กฎเหล็ก:\n"
            "1. **ห้ามพิมพ์ตารางด้วยตัวอักษร** (เช่น | ชื่อ | สกุล |) เพราะจะทำให้หน้าเว็บพัง ให้ใช้ Tag [SHOW_...] เท่านั้น\n"
            "2. ห้ามมั่วข้อมูลที่ไม่มีใน Context\n"
            "\n"
            f"=== DOCUMENT CONTEXT ===\n{context_text}\n========================"
        )
# -------------------------------------------------------------------
    # 4) Call LLM (Chain of Fallback: OpenRouter -> Google -> Raw)
    # -------------------------------------------------------------------
    
    answer_text = ""
    ai_response = None
    
    # [NEW] สร้าง Messages List สำหรับส่งให้ LLM โดยรวม History ด้วย
    messages = [SystemMessage(content=system_prompt)]
    
    for h in history[-6:]: 
        if h.get("role") == "user":
            messages.append(HumanMessage(content=h.get("content", "")))
        else:
            messages.append(AIMessage(content=h.get("content", "")))
    
    final_input = (
        f"คำถามปัจจุบัน: {query}\n\n"
        f"(หมายเหตุระบบ: Context สำหรับตอบคำถามนี้ถูกค้นหามาด้วย keyword: '{search_query}')"
    )
    messages.append(HumanMessage(content=final_input))
    
    # --- 1. แผน A: ลองใช้ Primary LLM ---
    try:
        if llm:
            ai_response = await llm.ainvoke(messages)
            answer_text = getattr(ai_response, "content", str(ai_response))
            
            # [NEW] 7. Fact Check (Self-Correction)
            if len(answer_text) > 100:
                answer_text = await _verify_answer_factuality(llm, query, answer_text, context_text)

    except Exception as e:
        logger.warning(f"[rag] ❌ Primary LLM failed: {e}")

    # --- 2. แผน B: ถ้าแผน A พัง ให้ลองใช้ Google Gemini ---
    if not answer_text or answer_text == "AI Error":
        try:
            google_llm = _get_google_llm()
            if google_llm:
                logger.info("[rag] 🔄 Switching to Backup LLM: Google Gemini...")
                ai_response = await google_llm.ainvoke(messages)
                answer_text = getattr(ai_response, "content", str(ai_response))
        except Exception as e_google:
             logger.error(f"[rag] ❌ Google LLM also failed: {e_google}")

    # --- 3. แผน C: ถ้าทุกอย่างพัง ---
    if not answer_text:
        if mode == "table" and found_table_ids:
             answer_text = "" 
        else:
             logger.warning("[rag] ⚠️ All LLMs failed. Using Raw Fallback.")
             return {
                 "answer": _generate_fallback_answer(docs, "System Error"), 
                 "sources": [], 
                 "intent": intent, 
                 "mode": f"{mode}+error"
             }

    # --- 5) Regex Replacement ---
    if (table_map or table_cat_map) and answer_text:
        try:
            # Replace Categories
            pattern_cat = re.compile(r"\[(?:SHOW_TABLE|SHOW|TABLE)[^:]*:\s*CAT\s*=\s*([^\]]+)\]", re.IGNORECASE)
            
            def replace_cat(match):
                cat_name = match.group(1).strip().lower()
                cat_key = f"cat:{cat_name}"
                if cat_key in table_cat_map: return f"\n<div class='my-4 overflow-x-auto border rounded-lg shadow-sm bg-white p-2'>{table_cat_map[cat_key]}</div>\n"
                role_key = f"role:{cat_name}"
                if role_key in table_cat_map: return f"\n<div class='my-4 overflow-x-auto border rounded-lg shadow-sm bg-white p-2'>{table_cat_map[role_key]}</div>\n"
                return match.group(0)
            
            answer_text = pattern_cat.sub(replace_cat, answer_text)
            
            # Replace TBL_x
            def replace_match(match):
                found_id = match.group(1).strip() # TBL_1 or 1
                clean_id = found_id.replace("TBL_", "") # 1

                raw_html = ""
                # Try finding TBL_1
                if found_id in table_map: raw_html = table_map[found_id]
                # Try finding 1 (by assuming TBL_ prefix)
                elif f"TBL_{clean_id}" in table_map: raw_html = table_map[f"TBL_{clean_id}"]
                # Try finding exact key if numeric
                elif clean_id in table_map: raw_html = table_map[clean_id]
                
                # ถ้าเจอข้อมูลตาราง ให้ทำการ "ล้างไพ่" (Clean Attributes)
                if raw_html:
                    clean_html = re.sub(r'<table[^>]*>', '<table>', raw_html, flags=re.IGNORECASE)
                    return f"\n<div class='answer-tables-content'>{clean_html}</div>\n"

                return match.group(0)

            pattern = re.compile(r"\[(?:SHOW_TABLE|SHOW|TABLE)[^:]*:\s*(?:TBL[_]?)?\s*([\w\d\.]+)\]", re.IGNORECASE)
            answer_text = pattern.sub(replace_match, answer_text)
            
        except Exception as e:
            logger.error(f"[rag] Regex replacement failed: {e}")

    # 6) Sources
    sources = []
    for d in docs:
        md = d.metadata or {}
        sources.append({
            "doc_id": md.get("doc_id"),
            "page": md.get("page"),
            "source": md.get("source"),
            "chunk_id": md.get("chunk_id")
        })

    return {"answer": answer_text, "sources": sources, "intent": intent, "mode": f"{mode}+qna_llm"}