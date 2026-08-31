"""
backend/services/contextual_chunker.py

Anthropic-style contextual chunking (MVP: doc-level context).
Before embedding, each chunk gets a short prefix describing what the whole
document is about + which page/source the chunk comes from.

This lets the embedding model + reranker + LLM disambiguate chunks that
contain generic content (e.g. "11 มี.ค. - 25 เม.ย." — is this
registration? training? announcement?).

For each doc, we do ONE LLM call to generate a 1-2 sentence summary of the
whole document from a sample of its chunks. The summary is then prepended
to every chunk from that doc before embedding.

Full per-chunk contextual generation (Anthropic's original) is much more
expensive (350+ LLM calls per re-ingest). This doc-level variant costs 1
LLM call per doc — often enough to unlock large recall wins.
"""
from __future__ import annotations

import os
import logging
from typing import List, Dict, Any
from collections import defaultdict

logger = logging.getLogger(__name__)


CONTEXTUAL_ENABLED = os.getenv("CONTEXTUAL_CHUNKING", "true").lower() not in ("false", "0", "no")
CONTEXT_SUMMARY_MODEL = os.getenv("CONTEXT_SUMMARY_MODEL", "qwen2.5:7b")
CONTEXT_SUMMARY_MAX_CHARS = int(os.getenv("CONTEXT_SUMMARY_MAX_CHARS", "3000"))

SUMMARY_PROMPT = """คุณจะได้เห็นตัวอย่างเนื้อหาจากเอกสาร โปรดสรุปเอกสารนี้อย่างสั้น (1-2 ประโยค, ไม่เกิน 40 คำ) ให้ชัดเจนว่าเอกสารนี้เป็น "ประเภทอะไร" และ "เกี่ยวกับหัวข้อใด" เช่น:
- "สัญญาจ้างเหมาบริการรายบุคคล ระหว่างมหาวิทยาลัยราชภัฏยะลา และ นายอมรเทพ มุขยวัฒน์"
- "รายงานสรุปโครงการอบรม AI สำหรับครู 3 จังหวัดชายแดนใต้ ปี 2569"

ตัวอย่างเนื้อหา:
{content}

สรุปในบรรทัดเดียว:"""


def _get_llm_client():
    """Return a configured OpenAI-compatible client for local Ollama LLM.
    Returns None on failure so caller can skip gracefully."""
    try:
        from ingestion.config import CUSTOM_API_BASE, CUSTOM_API_KEY
        from openai import OpenAI
        if not CUSTOM_API_BASE or not CUSTOM_API_KEY:
            return None
        return OpenAI(api_key=CUSTOM_API_KEY, base_url=CUSTOM_API_BASE)
    except Exception as e:
        logger.warning(f"[contextual] LLM client init failed: {e}")
        return None


def _generate_doc_summary(chunks_for_doc: List[Any]) -> str:
    """Sample the first few chunks, call LLM, return a 1-2 sentence summary.
    Empty string on failure — caller will skip prefixing."""
    if not chunks_for_doc:
        return ""

    client = _get_llm_client()
    if client is None:
        return ""

    # Build sample from first N chunks up to max chars
    parts = []
    total = 0
    for c in chunks_for_doc:
        t = (c.content or "").strip()
        if not t:
            continue
        snippet = t[:600]
        parts.append(snippet)
        total += len(snippet)
        if total >= CONTEXT_SUMMARY_MAX_CHARS:
            break
    sample = "\n".join(parts)[:CONTEXT_SUMMARY_MAX_CHARS]

    try:
        resp = client.chat.completions.create(
            model=CONTEXT_SUMMARY_MODEL,
            messages=[{"role": "user", "content": SUMMARY_PROMPT.format(content=sample)}],
            temperature=0.1,
            max_tokens=120,
        )
        summary = (resp.choices[0].message.content or "").strip()
        # Sanity: strip newlines / trim
        summary = " ".join(summary.split())[:300]
        return summary
    except Exception as e:
        logger.warning(f"[contextual] summary generation failed: {e}")
        return ""


def augment_chunks_with_context(chunks: List[Any]) -> List[Any]:
    """Prepend a doc-level context prefix to each chunk's `content` field.
    Original content is preserved in `metadata['orig_content']`.

    Returns the same list of chunks, mutated in-place.
    No-op if CONTEXTUAL_CHUNKING env is off, list is empty, or LLM unavailable.
    """
    if not CONTEXTUAL_ENABLED or not chunks:
        return chunks

    # Group by doc_id
    by_doc: Dict[str, List[Any]] = defaultdict(list)
    for c in chunks:
        by_doc[c.doc_id].append(c)

    # Generate summaries once per doc
    doc_summaries: Dict[str, str] = {}
    for doc_id, group in by_doc.items():
        summary = _generate_doc_summary(group)
        doc_summaries[doc_id] = summary
        logger.info(
            f"[contextual] doc={doc_id} · chunks={len(group)} · "
            f"summary={'(none)' if not summary else summary[:120]}"
        )

    # Prefix each chunk
    n_augmented = 0
    for c in chunks:
        summary = doc_summaries.get(c.doc_id, "")
        if not summary:
            continue

        page_hint = f" page={c.page}" if c.page is not None else ""
        src_hint = f" type={c.source}" if c.source else ""
        prefix = f"[Doc: {c.doc_id}]{page_hint}{src_hint}\n[Summary: {summary}]\n\n"

        # Preserve original before mutating
        if isinstance(c.metadata, dict):
            c.metadata = {**c.metadata, "orig_content": c.content}
        c.content = prefix + c.content
        n_augmented += 1

    logger.info(f"[contextual] augmented {n_augmented}/{len(chunks)} chunks across {len(by_doc)} docs")
    return chunks
