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
    
    header = f"⚠️ **แจ้งเตือน ({error_msg}):** ระบบจึงดึงเนื้อหาที่เกี่ยวข้องจากเอกสารมาแสดงให้โดยตรงครับ:\n\n"
             
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
    query_terms = set(query.lower().split())
    scored_docs = []
    for d in docs:
        content = (getattr(d, "page_content", "") or "").lower()
        keyword_score = 0
        for term in query_terms:
            if term in content:
                keyword_score += 1.5
        scored_docs.append({"doc": d, "score": keyword_score})

    # 2. Cross-Encoder Re-ranking (The Real Intelligence)
    reranker = _get_reranker_model()
    if reranker:
        try:
            pairs = [[query, d["doc"].page_content] for d in scored_docs]
            ai_scores = reranker.predict(pairs)
            for i, score in enumerate(ai_scores):
                scored_docs[i]["score"] += float(score)
            logger.debug(f"[rag] Re-ranking complete for {len(docs)} docs.")
        except Exception as e:
            logger.warning(f"[rag] Re-ranking failed (using basic sort): {e}")

    # 3. Sort & Cut
    scored_docs.sort(key=lambda x: x["score"], reverse=True)
    final_docs = [item["doc"] for item in scored_docs]
    return final_docs[:top_k]


# -------------------------------------------------------------------
# 4) Q&A extraction + matching utilities
# -------------------------------------------------------------------
def _load_qna_pairs_for_doc(doc_id: str) -> List[Dict[str, str]]:
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

    if not best_answer or best_score < 0.45:
        return None

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
# 5) main RAG function (IMPROVED & FIXED 500 ERROR)
# -------------------------------------------------------------------
# [FIXED & SAFE] แก้ไขให้ return dict พร้อม mode เสมอ กัน Error 500
async def answer_question(
    query: str,
    doc_ids: Optional[List[str]] = None,
    top_k: int = 10,
    mode: str = "auto",
) -> Dict:
    
    # 1. Input Check
    if not query or not query.strip():
        return {"answer": "คำถามว่าง", "sources": [], "intent": None, "mode": mode}

    if mode == "auto":
        intent = _rule_based_intent(query) or "text"
    elif mode in ("text", "table", "both"):
        intent = mode
    else:
        intent = "text"

    doc_types = None
    sources_filter = ["table", "text"] if intent == "table" else ["text", "table"]

    # 2. Search
    docs = []
    try:
        raw_docs = search_similar(query, k=top_k*3, doc_ids=doc_ids, sources=sources_filter, doc_types=doc_types)
        docs = _rerank_documents(query, raw_docs, top_k)
    except Exception as e:
        logger.error(f"Search failed: {e}")
        # [CRITICAL FIX] ใส่ mode กลับไปด้วย เพื่อไม่ให้ pydantic validate fail
        return {"answer": f"ระบบค้นหาขัดข้อง: {str(e)}", "sources": [], "intent": intent, "mode": mode}

    if not docs:
        return {"answer": "ไม่พบข้อมูลที่เกี่ยวข้อง", "sources": [], "intent": intent, "mode": mode}

    # 2.5 Deterministic Q&A
    try:
        qna_direct = _find_best_qna_answer_from_docs(query, docs)
        if qna_direct:
            return {"answer": qna_direct["answer"], "sources": qna_direct["sources"], "intent": intent, "mode": f"{mode}+qna"}
    except Exception:
        pass 

    # --- 3) Prepare Context & Table Map ---
    context_parts = []
    table_map = {} 
    table_counter = 0 
    
    try:
        for i, d in enumerate(docs, 1):
            md = d.metadata or {}
            
            # Normalize source
            source = str(md.get("source", "text")).lower().strip()
            
            doc_name = md.get("doc_id", "unknown")
            page = md.get("page", "?")
            
            if source == "table":
                table_counter += 1
                table_ref_id = str(table_counter)
                
                # Retrieve HTML content
                html_content = md.get("html_content", "")
                if not html_content:
                    html_content = f"<pre class='text-xs overflow-auto p-2 bg-gray-100'>{md.get('markdown_content', 'No content')}</pre>"
                
                table_map[table_ref_id] = html_content
                
                context_parts.append(
                    f"--- SOURCE {i} (Type: Table) ---\n"
                    f"Document: {doc_name}, Page: {page}\n"
                    f"Table ID: TBL_{table_ref_id}\n"
                    f"Content:\n{md.get('markdown_content', d.page_content)}\n"
                    f"-------------------------------------------"
                )
            else:
                context_parts.append(
                    f"--- SOURCE {i} (Type: Text) ---\n"
                    f"{d.page_content[:3000]}\n"
                )
    except Exception as e:
        logger.error(f"Context build failed: {e}")
        return {"answer": "เกิดข้อผิดพลาดในการเตรียมข้อมูล", "sources": [], "intent": intent, "mode": mode}

    context_text = "\n\n".join(context_parts)
    
    system_prompt = (
        "คุณเป็นผู้ช่วยอัจฉริยะที่ตอบคำถามจากเอกสาร\n"
        "กฎสำคัญ:\n"
        "1. ห้ามสร้างตารางด้วย Markdown (|...|) เด็ดขาด\n"
        "2. ถ้าคำตอบอ้างอิงข้อมูลจากตารางที่มี 'Table ID: TBL_x' ให้คุณ:\n"
        "   - สรุปใจความสำคัญเป็นข้อความสั้นๆ\n"
        "   - จบด้วย Tag: [SHOW_TABLE:TBL_x] ตาม ID ที่ระบุใน Context เท่านั้น\n"
        "3. ห้ามคิดข้อมูลเอง ให้ใช้ข้อมูลจาก Context เท่านั้น\n"
        "\n"
        f"=== CONTEXT ===\n{context_text}\n==============="
    )

    # 4) Call LLM
    llm = _get_llm_instance()
    if not llm:
        return {"answer": _generate_fallback_answer(docs, "No LLM"), "sources": [], "intent": intent, "mode": f"{mode}+no_llm"}

    answer_text = "AI Error"
    try:
        resp = await llm.ainvoke([SystemMessage(content=system_prompt), HumanMessage(content=query)])
        answer_text = getattr(resp, "content", str(resp))
    except Exception as e:
        logger.error(f"LLM Error: {e}")
        # [CRITICAL FIX] ใส่ mode กลับไปด้วย
        return {"answer": _generate_fallback_answer(docs, "Quota/Error"), "sources": [], "intent": intent, "mode": f"{mode}+error"}

    # --- 5) Regex Replacement ---
    if table_map and answer_text:
        try:
            def replace_match(match):
                # match.group(1) คือตัวเลขที่จับได้ (เช่น '1' จาก 'TBL_1' หรือ '4.2' จาก 'TBL_4.2')
                found_id = match.group(1)
                
                # 1. ลองหาแบบตรงๆ
                if found_id in table_map:
                    html = table_map[found_id]
                    return f"\n<div class='my-4 overflow-x-auto border rounded-lg shadow-sm bg-white p-2'>{html}</div>\n"
                
                # 2. ถ้าหาไม่เจอ ลองตัดทศนิยมทิ้ง (เผื่อ AI เอาเลขข้อมาตอบ เช่น 4.2 -> 4)
                if "." in found_id:
                    simple_id = found_id.split(".")[0]
                    if simple_id in table_map:
                         html = table_map[simple_id]
                         return f"\n<div class='my-4 overflow-x-auto border rounded-lg shadow-sm bg-white p-2'>{html}</div>\n"

                # 3. ถ้ายังไม่เจออีก ให้ลองดูว่าถ้ามีตารางเดียวใน Context ก็เอามาใส่เลย (Fallback สุดท้าย)
                if len(table_map) == 1:
                    first_key = list(table_map.keys())[0]
                    return f"\n<div class='my-4 overflow-x-auto border rounded-lg shadow-sm bg-white p-2'>{table_map[first_key]}</div>\n"

                return match.group(0)

            # Regex ที่ครอบคลุมทศนิยม: TBL1, TBL_1, TBL 1, TBL_4.2
            pattern = re.compile(r"\[(?:SHOW_TABLE|SHOW|TABLE)[^:]*:\s*TBL[_]?\s*([\d\.]+)\]", re.IGNORECASE)
            answer_text = pattern.sub(replace_match, answer_text)
        except Exception as e:
            logger.error(f"Regex replacement failed: {e}")

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

    # [CRITICAL] Return Dictionary พร้อม mode เสมอ ห้าม return None
    return {"answer": answer_text, "sources": sources, "intent": intent, "mode": f"{mode}+qna_llm"}