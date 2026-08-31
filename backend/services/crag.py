"""
backend/services/crag.py

Corrective RAG grader.

After retrieval + rerank, we ask a small LLM to grade whether the top-K
chunks actually contain the answer to the user's question. Three verdicts:

- "yes"     → the chunks answer the question directly → proceed normally
- "partial" → some info exists but not the full answer → answer with disclaimer
- "no"      → the chunks don't contain the answer → refuse (return "ไม่พบข้อมูล")

This reduces false-confident wrong answers where the LLM invents details from
loosely-related context.

Design:
- Uses a fast model (default qwen2.5:7b via Ollama) for the grader — separate
  from the main answer LLM so quality-latency trade-offs are decoupled
- One extra call per query (~5-15s on GPU); toggle via CRAG_ENABLED env
- Falls back to "yes" on any grader failure (never blocks a legit answer)
"""
from __future__ import annotations

import os
import re
import json
import logging
from typing import List, Dict, Any, Optional

logger = logging.getLogger(__name__)

CRAG_ENABLED = os.getenv("CRAG_ENABLED", "true").lower() not in ("false", "0", "no")
CRAG_MODEL = os.getenv("CRAG_MODEL", "qwen2.5:7b")
# How many top chunks to send to the grader — enough to give it fair shot
CRAG_TOP_K = int(os.getenv("CRAG_TOP_K", "5"))
# Cap chunk length so the grader prompt stays small
CRAG_CHUNK_MAX_CHARS = int(os.getenv("CRAG_CHUNK_MAX_CHARS", "800"))


GRADER_PROMPT = """คุณคือระบบตรวจสอบว่าบริบทมี "ความเกี่ยวข้อง" กับคำถามหรือไม่

คำถาม: {query}

บริบทที่ระบบดึงมา (chunks จากเอกสาร):
{chunks}

**เกณฑ์การตัดสิน (มีอคติเอนไปทาง yes/partial):**
- "yes"     = บริบทมีข้อมูลที่เกี่ยวข้องกับคำถามอย่างชัดเจน (อยู่ในหัวข้อเดียวกัน แม้ยังไม่ตอบตรงเป๊ะ)
- "partial" = บริบทมีข้อมูลเกี่ยวข้องบางส่วน หรืออาจต้องประมวลจากหลาย chunks
- "no"      = บริบท**ไม่เกี่ยวข้องเลย**กับคำถาม (คนละหัวข้อ/คนละเรื่อง)

**หลักการสำคัญ:**
- ถ้าเห็น keyword ในคำถามปรากฏใน chunk ใด ให้เป็น "yes" อย่างน้อย
- ถ้าคำถามอยู่ในโดเมนเดียวกับเอกสาร (แม้ตอบไม่ได้เต็ม) ให้ "partial"
- ตอบ "no" **เฉพาะเมื่อ**บริบทคนละเรื่องกับคำถาม 100%

ตอบเป็น JSON เท่านั้น (ห้ามคำอธิบายอื่น):
{{"verdict": "yes" | "partial" | "no", "reason": "..."}}"""


def _format_chunks_for_grader(chunks: List[Dict[str, Any]]) -> str:
    """Format top chunks into a compact text block for the grader prompt."""
    parts = []
    for i, c in enumerate(chunks[:CRAG_TOP_K], start=1):
        text = (c.get("content") or "").strip()[:CRAG_CHUNK_MAX_CHARS]
        parts.append(f"[Source {i}] {text}")
    return "\n\n".join(parts)


def _parse_verdict(text: str) -> Dict[str, Any]:
    """Extract the JSON verdict, tolerant to markdown fences / prose."""
    m = re.search(r"\{[^{}]*\}", text, re.DOTALL)
    if not m:
        return {"verdict": "yes", "reason": "parse-fail"}
    try:
        d = json.loads(m.group(0))
        verdict = str(d.get("verdict", "yes")).lower().strip()
        if verdict not in ("yes", "partial", "no"):
            verdict = "yes"
        return {"verdict": verdict, "reason": d.get("reason", "")}
    except Exception:
        return {"verdict": "yes", "reason": "parse-fail"}


def grade_retrieval(query: str, chunks: List[Dict[str, Any]]) -> Dict[str, Any]:
    """LLM-graded relevance check. Returns dict with verdict + reason.

    On any failure (LLM down, parse error, disabled), returns {"verdict": "yes"}
    so answering proceeds — never blocks a legit response.
    """
    if not CRAG_ENABLED or not chunks:
        return {"verdict": "yes", "reason": "disabled or no chunks"}

    try:
        import litellm
        from ingestion.config import CUSTOM_API_BASE, CUSTOM_API_KEY
    except Exception as e:
        logger.warning(f"[crag] litellm import failed: {e}")
        return {"verdict": "yes", "reason": f"import-fail: {e}"}

    prompt = GRADER_PROMPT.format(
        query=query,
        chunks=_format_chunks_for_grader(chunks),
    )

    try:
        resp = litellm.completion(
            model=f"openai/{CRAG_MODEL}",
            messages=[{"role": "user", "content": prompt}],
            api_base=CUSTOM_API_BASE,
            api_key=CUSTOM_API_KEY,
            temperature=0.0,
            max_tokens=200,
            timeout=30,
        )
        text = (resp.choices[0].message.content or "").strip()
    except Exception as e:
        logger.warning(f"[crag] grader call failed: {e}")
        return {"verdict": "yes", "reason": f"call-fail: {e}"}

    result = _parse_verdict(text)
    logger.info(f"[crag] verdict={result['verdict']} reason={result.get('reason', '')[:100]}")
    return result
