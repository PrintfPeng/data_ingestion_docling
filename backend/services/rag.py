from __future__ import annotations

import os
import re
import json
import logging
import math
from typing import Dict, List, Optional
from pathlib import Path
from difflib import SequenceMatcher

from dotenv import load_dotenv

# LangChain / Gemini chat wrapper used in project
try:
    from langchain_google_genai import ChatGoogleGenerativeAI
    from langchain_core.messages import HumanMessage, SystemMessage
    _HAS_GENAI = True
except Exception:
    ChatGoogleGenerativeAI = None  # type: ignore
    HumanMessage = None  # type: ignore
    SystemMessage = None  # type: ignore
    _HAS_GENAI = False

# --- Re-ranking imports ---
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
_QNA_CACHE_MAX_SIZE = 100

# load .env to make sure key available
load_dotenv(override=True)
_GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

# LLM defaults
_LL_MODEL_FAST = "gemini-2.5-flash" 
_LL_MODEL_SMALL = "gemini-2.5-flash-lite"
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
# [NEW FIX] Sanitize Document ID (ให้ตรงกับ Backend)
# -------------------------------------------------------------------
def sanitize_doc_id(doc_id: str) -> str:
    """
    Sanitize document ID to match backend storage format.
    """
    if not doc_id:
        return ""
    # Lowercase
    doc_id = doc_id.lower().strip()
    # Replace spaces with underscores
    doc_id = re.sub(r'\s+', '_', doc_id)
    # Remove special characters (keep only alphanumeric and underscore)
    doc_id = re.sub(r'[^a-z0-9_]', '', doc_id)
    return doc_id


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
# Helper: LLM (Gemini) safe getter
# -------------------------------------------------------------------
def _get_llm_instance(model: Optional[str] = None, temperature: float = _DEFAULT_TEMPERATURE):
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
            max_retries=2,
            request_timeout=30
        )
    except Exception as e:
        logger.exception("[rag] Failed to init LLM: %s", e)
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
# [NEW] Intent & Logic Guards (เพิ่มส่วนนี้)
# -------------------------------------------------------------------

def _rule_based_intent(query: str) -> Optional[str]:
    # (ฟังก์ชันเดิม เก็บไว้ใช้เป็น Helper แต่ logic หลักย้ายไป auto mode selector)
    if not query or not query.strip(): return None
    q = query.lower()
    table_keywords = ["ตาราง", "table", "คอลัมน์", "column", "แถว", "row", "สรุป", "summary", "ยอด", "amount", "list", "รายการ", "schedule"]
    image_keywords = ["รูป", "รูปภาพ", "image", "logo", "กราฟ", "graph", "chart", "diagram", "photo", "ภาพ"]
    is_table = any(w in q for w in table_keywords)
    is_image = any(w in q for w in image_keywords)
    if is_table and not is_image: return "table"
    if is_image and not is_table: return "both"
    if is_table and is_image: return "both"
    return "text"

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
    # แยกคำง่ายๆ (สำหรับภาษาไทยควรใช้ PyThaiNLP แต่ใช้ split space/common chars เบื้องต้น)
    q_clean = re.sub(r'[^\w\s]', '', query).lower()
    t_clean = re.sub(r'[^\w\s]', '', text).lower()
    
    q_tokens = set(q_clean.split())
    t_tokens = set(t_clean.split())
    
    # ตัด Stopwords ทั่วไปออก
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
    MAX_TOKENS_ESTIMATE = 12000

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
# [NEW] Filter Table Documents by Category/Role
# -------------------------------------------------------------------
def _filter_table_docs_by_category(docs, query: str):
    return docs


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
    query_terms = query.lower().split()
    scored_docs = []
    for d in docs:
        content = (getattr(d, "page_content", "") or "").lower()
        base_score = 0.0
        
        for term in query_terms:
            if term in content:
                base_score += 1.0
        
        if query.lower() in content:
            base_score += 3.0
            
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
                    scored_docs[idx].metadata["ai_score"] = norm_score
                    scored_docs[idx].metadata["raw_score"] = float(raw)
                
                # Sort by AI Score
                scored_docs.sort(key=lambda x: x.metadata["ai_score"], reverse=True)
                return scored_docs[:top_k]

        except Exception as e:
            logger.warning(f"[rag] Re-ranking failed: {e}")

    # 3. Sort & Cut (Fallback to keyword score)
    scored_docs.sort(key=lambda x: x.metadata.get("keyword_score", 0), reverse=True)
    # Assign dummy confidence for filter to work
    for d in scored_docs:
        if d.metadata["ai_score"] == 0.0:
            d.metadata["ai_score"] = 0.3 # Dummy score to pass filter if keyword matches
    
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
    
    reranker = _get_reranker_model()
    if reranker:
        try:
            input_pairs = [[query, p["question"]] for p in all_pairs]
            raw_scores = reranker.predict(input_pairs)
            
            for i, raw in enumerate(raw_scores):
                norm_score = normalize_score(float(raw))
                if norm_score > best_score:
                    best_score = norm_score
                    best_item = all_pairs[i]
        except Exception:
            pass

    if not best_item:
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
# 5) main RAG function (UPGRADED & ROBUST)
# -------------------------------------------------------------------
async def answer_question(
    query: str,
    doc_ids: Optional[List[str]] = None,
    top_k: int = 10,
    mode: str = "auto",
) -> Dict:
    
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

    # [NEW] STEP 2: Mode Selection (Deterministic)
    if mode == "auto":
        q_lower = query.lower()
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

    doc_types = None
    sources_filter = None 

    # 3. Search (3-Layer Fallback Strategy)
    docs = []
    raw_docs = []

    try:
        # Layer 1: Strict Search
        raw_docs = search_similar(query, k=top_k*3, doc_ids=sanitized_doc_ids, sources=sources_filter, doc_types=doc_types)
        
        # Layer 2: Relaxed ID Search
        if not raw_docs and sanitized_doc_ids:
             logger.warning(f"[rag] Layer 1 empty. Retrying Layer 2 (Global search).")
             raw_docs = search_similar(query, k=top_k*3, doc_ids=None, sources=sources_filter, doc_types=doc_types)
        
        # Layer 3: Keyword/Broad Search
        if not raw_docs:
             logger.warning(f"[rag] Layer 2 empty. Retrying Layer 3 (Broad).")
             broad_docs = search_similar(query, k=50, doc_ids=None, sources=sources_filter, doc_types=doc_types)
             
             q_terms = [t for t in query.lower().split() if len(t) > 2]
             if q_terms:
                 raw_docs = [d for d in broad_docs if any(t in (d.page_content or "").lower() for t in q_terms)]
                 if not raw_docs: raw_docs = broad_docs[:top_k]
             else:
                 raw_docs = broad_docs[:top_k]

        logger.info(f"[rag] Found {len(raw_docs)} raw docs")

        # Rerank
        docs = _rerank_documents(query, raw_docs, top_k)
        
        # [NEW] STEP 4: STRICT FILTERING (No Rescue Mission)
        relevant_docs = _filter_relevant_docs(query, docs, min_score=MIN_SCORE_THRESHOLD)
        
        if not relevant_docs:
            # Check Q&A direct match before giving up
            qna_match = _find_best_qna_answer_from_docs(query, docs) # Use original docs to find doc_id context
            if qna_match:
                return {
                    "answer": qna_match["answer"],
                    "sources": qna_match["sources"],
                    "intent": "qna",
                    "mode": f"{mode}+qna"
                }
            
            return {
                "answer": "ไม่พบข้อมูลที่ตรงกับคำถามในเอกสารที่แนบมาครับ (Relevance Score ต่ำเกินไป)",
                "sources": [],
                "intent": intent,
                "mode": mode
            }
            
        docs = relevant_docs # Use filtered docs

    except Exception as e:
        logger.error(f"[rag] Search failed: {e}")
        return {"answer": f"ระบบค้นหาขัดข้อง: {str(e)}", "sources": [], "intent": intent, "mode": mode}

    # --- Prepare Context & Table Map ---
    table_map = {}
    table_cat_map = {}
    table_counter = 0 
    
    try:
        context_text = _build_context_text(docs)
        
        for i, d in enumerate(docs, 1):
            md = d.metadata or {}
            source = str(md.get("source", "text")).lower().strip()
            
            if source == "table":
                table_counter += 1
                table_ref_id = str(table_counter)
                
                raw_html = md.get("html_content", "")
                safe_html = _sanitize_html_content(raw_html)
                if not safe_html:
                    safe_html = f"<pre class='text-xs overflow-auto p-2 bg-gray-100'>{md.get('markdown_content', 'No content')}</pre>"
                
                table_map[table_ref_id] = safe_html
                
                category = md.get("category", "").strip().lower()
                role = md.get("role", "").strip().lower()
                
                if category:
                    cat_key = f"cat:{category}"
                    if cat_key not in table_cat_map: table_cat_map[cat_key] = safe_html
                if role:
                    role_key = f"role:{role}"
                    if role_key not in table_cat_map: table_cat_map[role_key] = safe_html

    except Exception as e:
        logger.error(f"[rag] Context build failed: {e}")
        return {"answer": "เกิดข้อผิดพลาดในการเตรียมข้อมูล", "sources": [], "intent": intent, "mode": mode}
    
    system_prompt = (
        "คุณเป็นผู้ช่วยอัจฉริยะ (Smart Assistant) ที่ตอบคำถามจากเอกสารที่กำหนดให้ (Context) เท่านั้น\n"
        "คำแนะนำ:\n"
        "1. **วิเคราะห์คำถาม:** จับประเด็นสำคัญว่าผู้ใช้ต้องการทราบอะไร\n"
        "2. **ค้นหาข้อมูล:** ดูข้อมูลใน Context ว่ามีส่วนไหนที่ตอบคำถามได้บ้าง\n"
        "   - ⚠️ **สำคัญ:** แหล่งข้อมูลใน Context เรียงตามลำดับความสำคัญแล้ว ให้ความสำคัญกับลำดับต้นๆ ก่อน\n"
        "3. **สังเคราะห์คำตอบ:** เรียบเรียงคำตอบให้น่าอ่าน เป็นภาษาไทยที่สละสลวย\n"
        "4. **การแสดงตาราง (ยืดหยุ่น):**\n"
        "   - [SHOW_TABLE:TBL_x] สำหรับเรียกตารางด้วยเลข (เช่น TBL_1)\n"
        "   - [SHOW_TABLE:CAT=หมวดหมู่] สำหรับเรียกตารางด้วยหมวด\n"
        "5. **[สำคัญ] การใช้ข้อมูลจากตาราง:**\n"
        "   - หากข้อมูลคำตอบอยู่ในส่วนที่เป็น Table (SOURCE Type: Table) **ต้อง** ดึงข้อมูลนั้นมาตอบ อย่าบอกว่าไม่พบข้อมูล\n"
        "   - ถ้าตารางมีข้อมูลที่ตอบคำถามได้ ให้ตอบฟันธงไปเลย\n"
        "\n"
        "กฎเหล็ก:\n"
        "- ห้ามตอบนอกเหนือจากข้อมูลที่มีใน Context\n"
        "- ถ้าไม่พบข้อมูล ให้ตอบว่า 'ไม่พบข้อมูลในเอกสารที่แนบมา'\n"
        "- ห้ามแต่งเติมข้อมูลเอง (Anti-Hallucination)\n"
        "\n"
        f"=== CONTEXT ===\n{context_text}\n==============="
    )

    # 4) Call LLM
    llm = _get_llm_instance(model=_LL_MODEL_FAST)
    
    answer_text = "AI Error"
    try:
        if not llm: raise Exception("Primary LLM not configured")
        
        resp = await llm.ainvoke([SystemMessage(content=system_prompt), HumanMessage(content=query)])
        answer_text = getattr(resp, "content", str(resp))
        
    except Exception as e:
        logger.warning(f"[rag] Primary LLM ({_LL_MODEL_FAST}) failed: {e}")
        try:
            llm_fallback = _get_llm_instance(model=_LL_MODEL_SMALL)
            if llm_fallback:
                resp = await llm_fallback.ainvoke([SystemMessage(content=system_prompt), HumanMessage(content=query)])
                answer_text = getattr(resp, "content", str(resp))
            else:
                raise e 
        except Exception as e2:
            logger.error(f"[rag] Fallback LLM failed: {e2}")
            return {"answer": _generate_fallback_answer(docs, "Quota/Error"), "sources": [], "intent": intent, "mode": f"{mode}+error"}

    # --- 5) Regex Replacement ---
    if (table_map or table_cat_map) and answer_text:
        try:
            pattern_cat = re.compile(r"\[(?:SHOW_TABLE|SHOW|TABLE)[^:]*:\s*CAT\s*=\s*([^\]]+)\]", re.IGNORECASE)
            
            def replace_cat(match):
                cat_name = match.group(1).strip().lower()
                cat_key = f"cat:{cat_name}"
                if cat_key in table_cat_map: return f"\n<div class='my-4 overflow-x-auto border rounded-lg shadow-sm bg-white p-2'>{table_cat_map[cat_key]}</div>\n"
                role_key = f"role:{cat_name}"
                if role_key in table_cat_map: return f"\n<div class='my-4 overflow-x-auto border rounded-lg shadow-sm bg-white p-2'>{table_cat_map[role_key]}</div>\n"
                return match.group(0)
            
            answer_text = pattern_cat.sub(replace_cat, answer_text)
            
            def replace_match(match):
                found_id = match.group(1)
                if found_id in table_map: return f"\n<div class='my-4 overflow-x-auto border rounded-lg shadow-sm bg-white p-2'>{table_map[found_id]}</div>\n"
                if "." in found_id:
                    simple_id = found_id.split(".")[0]
                    if simple_id in table_map: return f"\n<div class='my-4 overflow-x-auto border rounded-lg shadow-sm bg-white p-2'>{table_map[simple_id]}</div>\n"
                if len(table_map) == 1:
                    first_key = list(table_map.keys())[0]
                    return f"\n<div class='my-4 overflow-x-auto border rounded-lg shadow-sm bg-white p-2'>{table_map[first_key]}</div>\n"
                return match.group(0)

            pattern = re.compile(r"\[(?:SHOW_TABLE|SHOW|TABLE)[^:]*:\s*TBL[_]?\s*([\d\.]+)\]", re.IGNORECASE)
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