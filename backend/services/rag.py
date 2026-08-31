# backend/services/rag.py
import os
import litellm
from typing import List, Optional, Dict, Any
from langchain_core.documents import Document

from .vector_store import search_similar, get_vector_store
from .reranker import rerank, is_available as reranker_available
from .hybrid_search import hybrid_search, hybrid_search_multi, is_available as hybrid_available
from .query_rewriter import rewrite_query, classify_query_intent
from .crag import grade_retrieval, CRAG_ENABLED
from .agentic_rag import plan_query, AGENTIC_ENABLED
from .model_router import resolve_llm, format_model_id, LLM_MODE_AUTO
from ingestion.config import CUSTOM_API_BASE, CUSTOM_API_KEY, CUSTOM_MODEL_NAME

QUERY_REWRITE_ENABLED = os.getenv("QUERY_REWRITE_ENABLED", "true").lower() not in ("false", "0", "no")
# Metadata (intent) filter: after rerank, boost chunks whose intent matches
# the query's classified intent — nudges the right topic to the top-K.
INTENT_FILTER_ENABLED = os.getenv("INTENT_FILTER_ENABLED", "true").lower() not in ("false", "0", "no")
# How much to add to rerank_score for an intent match (0 disables boost effect).
INTENT_BOOST = float(os.getenv("INTENT_BOOST", "0.3"))

# How many chunks to retrieve before reranking; reranker keeps top_k
RETRIEVE_TOP_K = int(os.getenv("RETRIEVE_TOP_K", "30"))
# Drop chunks whose rerank score is below this — filters out irrelevant docs
# BGE-reranker-v2-m3 scores usually 0-1 for relevant, negative for irrelevant
MIN_RERANK_SCORE = float(os.getenv("MIN_RERANK_SCORE", "0.05"))

# Small-to-Big / Sentence-Window: after rerank, expand each match to include
# neighboring chunks from the same page so LLM sees surrounding context.
PAGE_WINDOW_ENABLED = os.getenv("PAGE_WINDOW_ENABLED", "true").lower() not in ("false", "0", "no")
PAGE_WINDOW_MAX_CHARS = int(os.getenv("PAGE_WINDOW_MAX_CHARS", "3000"))


def _strip_contextual_prefix(text: str) -> str:
    """Contextual chunker prepends "[Doc: ...] ...\n[Summary: ...]\n\n<orig>"
    to chunks before embedding. Strip that prefix before showing to LLM so
    the same summary doesn't appear 5 times in the context window.
    """
    if not text or not text.startswith("[Doc:"):
        return text
    # First "\n\n" separates prefix from original content
    sep = text.find("\n\n")
    return text[sep + 2:] if sep != -1 else text


def _docs_to_sources(docs) -> List[Dict[str, Any]]:
    """แปลง langchain Document เป็น dict ที่ frontend/main.py ใช้งานต่อได้"""
    sources = []
    for d in docs:
        md = dict(d.metadata or {})
        # Prefer stored original if present, else strip prefix from embed-text
        original = md.get("orig_content") or _strip_contextual_prefix(d.page_content)
        sources.append({
            "content": original,
            "doc_id": md.get("doc_id"),
            "page": md.get("page"),
            "source": md.get("source", "text"),
            "metadata": md,
        })
    return sources


def _expand_to_page_window(docs: List[Document]) -> List[Document]:
    """Small-to-Big: for each doc, fetch all chunks with the same (doc_id, page)
    from Chroma and merge them into a single "page window" for LLM context.

    Places the originally-matched chunk at the top of the window, then appends
    other chunks from the same page (deduped). Caps window at PAGE_WINDOW_MAX_CHARS.
    """
    if not docs:
        return docs
    try:
        vectordb = get_vector_store()
        coll = vectordb._collection
    except Exception:
        return docs

    seen_pages = set()
    out: List[Document] = []
    for d in docs:
        md = d.metadata or {}
        doc_id = md.get("doc_id")
        page = md.get("page")
        key = (doc_id, page)

        # Only expand once per (doc, page) — subsequent chunks from same page
        # are captured by the window we already returned
        if key in seen_pages:
            continue
        seen_pages.add(key)

        # Can't expand without doc_id + page
        if not doc_id or page is None:
            out.append(d)
            continue

        try:
            raw = coll.get(
                where={"$and": [{"doc_id": doc_id}, {"page": page}]},
                include=["documents", "metadatas"],
            )
        except Exception:
            out.append(d)
            continue

        neighbor_texts = raw.get("documents", []) or []
        if len(neighbor_texts) <= 1:
            out.append(d)
            continue

        # Start with the originally-matched chunk, then append others
        matched_text = d.page_content
        merged_parts: List[str] = [matched_text]
        total = len(matched_text)
        matched_stripped = matched_text.strip()
        for text in neighbor_texts:
            if not text or text.strip() == matched_stripped:
                continue
            if total + len(text) > PAGE_WINDOW_MAX_CHARS:
                break
            merged_parts.append(text.strip())
            total += len(text)

        merged = "\n\n".join(merged_parts)
        new_doc = Document(page_content=merged, metadata=md)
        out.append(new_doc)

    return out


def _prepare_context_and_messages(
    query: str,
    doc_ids: Optional[List[str]] = None,
    top_k: int = 5,
    history: Optional[List[Dict[str, str]]] = None,
):
    """Shared retrieval + prompt building for both non-streaming and streaming
    answer paths. Returns (search_results, messages)."""
    # 1a. Retrieve top-N candidates — hybrid (BM25 + vector) + multi-query
    retrieve_k = max(top_k, RETRIEVE_TOP_K) if reranker_available() else top_k

    # Agentic planner: for complex multi-aspect queries, decompose into
    # sub-questions and retrieve for each — then merge before rerank.
    all_queries: List[str] = [query]
    if AGENTIC_ENABLED:
        plan = plan_query(query)
        if plan.get("needs_planning"):
            subs = plan.get("subqueries", [])
            # Keep the original query too so we still cover direct matches
            all_queries = [query] + [s for s in subs if s]

    # Expand each planned query with rule/LLM rewrites, then hybrid-search all
    if hybrid_available():
        if QUERY_REWRITE_ENABLED:
            expanded: List[str] = []
            for q in all_queries:
                for v in rewrite_query(q):
                    if v not in expanded:
                        expanded.append(v)
            docs = hybrid_search_multi(queries=expanded, k=retrieve_k, doc_ids=doc_ids)
        else:
            docs = hybrid_search_multi(queries=all_queries, k=retrieve_k, doc_ids=doc_ids)
    else:
        docs = search_similar(query=query, k=retrieve_k, doc_ids=doc_ids)

    # 1b. Rerank down to top_k using cross-encoder
    if reranker_available() and len(docs) > 1:
        # Over-fetch here so intent boost has more candidates to reorder
        pool_k = max(top_k * 2, top_k + 3) if INTENT_FILTER_ENABLED else top_k
        docs = rerank(query, docs, top_k=pool_k)
        if MIN_RERANK_SCORE > -999 and len(docs) > 1:
            filtered = [d for d in docs if d.metadata.get("rerank_score", 1.0) >= MIN_RERANK_SCORE]
            if filtered:
                docs = filtered

    # 1c. Intent-based reordering: chunks whose stored intent matches the
    # query's classified intent get a score boost so they float to the top.
    if INTENT_FILTER_ENABLED and len(docs) > 1:
        q_intent = classify_query_intent(query)
        if q_intent:
            for d in docs:
                chunk_intents = (d.metadata.get("intent") or "").lower()
                primary = (d.metadata.get("primary_intent") or "").lower()
                if q_intent in chunk_intents or q_intent == primary:
                    d.metadata["_intent_matched"] = True
                    d.metadata["rerank_score"] = d.metadata.get("rerank_score", 0.5) + INTENT_BOOST
            docs = sorted(docs, key=lambda d: d.metadata.get("rerank_score", 0.0), reverse=True)
        # Trim back to top_k after boost
        docs = docs[:top_k]

    # 1d. Small-to-Big / Sentence-Window: expand each match to full-page window
    if PAGE_WINDOW_ENABLED:
        docs = _expand_to_page_window(docs)

    search_results = _docs_to_sources(docs)

    context_parts = []
    for i, res in enumerate(search_results, start=1):
        did = res.get("doc_id") or "unknown"
        page = res.get("page")
        header = f"[Source {i}] doc={did}" + (f" page={page}" if page else "")
        context_parts.append(f"{header}\n{res['content']}")
    context_text = "\n\n---\n\n".join(context_parts)

    system_prompt = _build_system_prompt(context_text)
    messages = [{"role": "system", "content": system_prompt}]
    if history:
        messages.extend(history)
    messages.append({"role": "user", "content": query})
    return search_results, messages


def _build_system_prompt(context_text: str) -> str:
    return f"""คุณคือผู้ช่วยอัจฉริยะ ตอบคำถามจากบริบทที่ให้ไว้เท่านั้น
หากในบริบทไม่มีคำตอบ ให้บอกว่าไม่ทราบ ห้ามเดา
ใช้ภาษาไทยที่สุภาพและเป็นทางการ

**ข้อสำคัญ — การเทียบตัวเลข/วันที่:**
- เลขไทย ๐๑๒๓๔๕๖๗๘๙ เท่ากับ 0123456789 (เช่น ๕ = 5 = ห้า)
- คำภาษาไทย: ศูนย์=0, หนึ่ง=1, สอง=2, สาม=3, สี่=4, ห้า=5, หก=6, เจ็ด=7, แปด=8, เก้า=9, สิบ=10
- ปี พ.ศ. ต่างจาก ค.ศ. 543 ปี (พ.ศ. ๒๕๖๙ = ค.ศ. 2026)
- เมื่อคำถามใช้รูปแบบใดรูปแบบหนึ่ง ให้จับคู่กับข้อมูลในบริบทได้ทุกรูปแบบ

**การตีความคำถาม:**
- "ลงนามเมื่อวันที่เท่าใด" / "ทำสัญญาเมื่อไหร่" → ให้ดูวันที่ที่ปรากฏหลังคำว่า "ทำขึ้น ณ ... เมื่อวันที่ ..."
- "งวดที่ N" ในเอกสารอาจเขียนเป็น "งวดที่ N", "งวดที่ ๕", หรือใช้คำ (เช่น "งวดที่ห้า") — ให้ถือว่าเท่ากัน

**สำคัญเรื่องการเลือก Source:**
- บริบทมีหลาย Sources แต่ละอันมี header "[Source N] doc=... page=..."
- **Source 1 คือ chunk ที่ระบบคัดว่าเกี่ยวข้องที่สุด — ให้พิจารณา Source 1 ก่อน**
- Source 2+ ใช้เป็นข้อมูลเสริม/ยืนยัน หรือใช้เมื่อ Source 1 ไม่พอ
- ถ้าคำตอบอยู่ใน Source 1 อย่างชัดเจน ให้ตอบจาก Source 1 (ไม่ต้องไปมองที่อื่น)
- **ห้ามเดา** — ถ้าไม่มี Source ไหนตอบตรงคำถามได้ ให้บอกว่า "ไม่พบข้อมูลในเอกสาร" อย่าประกอบคำตอบเอง
- ห้ามเอาข้อมูลจาก Source ที่ **doc_id ต่างกัน** มารวมกัน เว้นแต่คำถามจะเปรียบเทียบข้าม doc ชัดเจน

บริบท (Context):
{context_text}
"""


async def answer_question(
    query: str,
    doc_ids: Optional[List[str]] = None,
    top_k: int = 5,
    mode: str = "auto",
    history: Optional[List[Dict[str, str]]] = None,
    llm_mode: str = LLM_MODE_AUTO,
) -> Dict[str, Any]:
    """Non-streaming: run retrieval + generation, return full answer.
    `llm_mode` picks which answer LLM to call (local Ollama vs cloud API).
    """
    search_results, messages = _prepare_context_and_messages(query, doc_ids, top_k, history)

    # Corrective RAG grader — refuse if chunks clearly don't answer the question
    if CRAG_ENABLED:
        verdict = grade_retrieval(query, search_results)
        if verdict.get("verdict") == "no":
            return {
                "answer": "ไม่พบข้อมูลในเอกสารที่จะตอบคำถามนี้ได้ กรุณาลองถามใหม่ด้วยคำอื่นหรือเพิ่ม doc filter",
                "sources": search_results,
                "intent": "refused_by_crag",
                "mode": mode,
                "llm_mode": llm_mode,
                "crag_reason": verdict.get("reason", ""),
            }

    llm_cfg = resolve_llm(llm_mode)
    try:
        response = litellm.completion(
            model=format_model_id(llm_cfg),
            messages=messages,
            api_base=llm_cfg.api_base,
            api_key=llm_cfg.api_key,
            temperature=0.2,
        )
        answer = response.choices[0].message.content
        return {
            "answer": answer,
            "sources": search_results,
            "intent": "rag_query",
            "mode": mode,
            "llm_mode": llm_mode,
            "llm_provider": llm_cfg.provider,
            "llm_model": llm_cfg.model,
        }
    except Exception as e:
        print(f"⚠️ [RAG-Error] AI Failed: {e}")
        return {
            "answer": f"ขออภัย เกิดข้อผิดพลาดในการเชื่อมต่อกับ AI: {str(e)}",
            "sources": [],
            "intent": "error",
            "mode": mode,
            "llm_mode": llm_mode,
            "llm_provider": llm_cfg.provider,
        }


async def answer_question_stream(
    query: str,
    doc_ids: Optional[List[str]] = None,
    top_k: int = 5,
    mode: str = "auto",
    history: Optional[List[Dict[str, str]]] = None,
    llm_mode: str = LLM_MODE_AUTO,
):
    """Streaming: async generator yielding (event_name, payload) tuples.
    Events:
      - "sources" (dict): retrieved sources — emitted immediately
      - "token" (dict): {"text": partial_content} — one per token chunk
      - "done"  (dict): final metadata {intent, mode}
      - "error" (dict): {message} on failure
    """
    try:
        search_results, messages = _prepare_context_and_messages(query, doc_ids, top_k, history)
    except Exception as e:
        yield ("error", {"message": f"retrieval failed: {e}"})
        return

    llm_cfg = resolve_llm(llm_mode)
    yield ("sources", {
        "sources": search_results, "mode": mode,
        "llm_provider": llm_cfg.provider, "llm_model": llm_cfg.model,
    })

    # CRAG grader for streaming path too
    if CRAG_ENABLED:
        verdict = grade_retrieval(query, search_results)
        if verdict.get("verdict") == "no":
            msg = "ไม่พบข้อมูลในเอกสารที่จะตอบคำถามนี้ได้"
            yield ("token", {"text": msg})
            yield ("done", {"intent": "refused_by_crag", "mode": mode, "llm_mode": llm_mode})
            return

    try:
        # Async streaming — doesn't block the event loop between tokens
        stream = await litellm.acompletion(
            model=format_model_id(llm_cfg),
            messages=messages,
            api_base=llm_cfg.api_base,
            api_key=llm_cfg.api_key,
            temperature=0.2,
            stream=True,
        )
        async for chunk in stream:
            try:
                delta = chunk.choices[0].delta.content or ""
            except (AttributeError, IndexError):
                delta = ""
            if delta:
                yield ("token", {"text": delta})
    except Exception as e:
        print(f"⚠️ [RAG-Stream-Error] {e}")
        yield ("error", {"message": str(e)})
        return

    yield ("done", {
        "intent": "rag_query", "mode": mode, "llm_mode": llm_mode,
        "llm_provider": llm_cfg.provider, "llm_model": llm_cfg.model,
    })
