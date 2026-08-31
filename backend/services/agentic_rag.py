"""
backend/services/agentic_rag.py

Lightweight Agentic RAG — query decomposition + multi-hop retrieval.

Compared to plain single-shot RAG, this pattern:
1. Sends the user query through a small **planner** LLM that decides:
   - Is the query simple (single-fact lookup)? → skip planning
   - Is the query complex (comparison, multi-aspect, cross-doc)? → produce 2-4 sub-questions
2. For each sub-question, run the normal hybrid_search + rerank pipeline
3. Merge all retrieved chunks (dedupe by chunk_id)
4. Hand the merged pool + original query to the answer LLM

Design principles:
- Fail-open: any planner failure just runs the original query — never blocks
- Cheap: uses a fast model (qwen2.5:7b default) for planning
- Optional: env AGENTIC_ENABLED gates the entire path
"""
from __future__ import annotations

import os
import re
import json
import logging
from typing import List, Dict, Any, Optional

logger = logging.getLogger(__name__)

AGENTIC_ENABLED = os.getenv("AGENTIC_ENABLED", "true").lower() not in ("false", "0", "no")
AGENTIC_MODEL = os.getenv("AGENTIC_MODEL", "qwen2.5:7b")
# Max sub-questions the planner may emit
AGENTIC_MAX_SUBQ = int(os.getenv("AGENTIC_MAX_SUBQ", "3"))
# Minimum query complexity to trigger planning (char count heuristic)
AGENTIC_MIN_QUERY_LEN = int(os.getenv("AGENTIC_MIN_QUERY_LEN", "15"))


# Keywords that hint at multi-aspect / cross-doc / comparison queries
_COMPLEX_HINTS = re.compile(
    r"(?:และ|กับ|เทียบ|เปรียบเทียบ|ต่างกัน|ทั้ง.+และ|มีอะไรบ้าง|กี่ประเภท|"
    r"vs|and|compare|both|multiple)",
    re.IGNORECASE,
)


PLANNER_PROMPT = """คุณคือระบบวางแผน RAG search (planner)

**หน้าที่:** ประเมินคำถามและตัดสินใจว่าต้องแยกค้นหลายครั้งหรือไม่

**คำถาม:** {query}

**เกณฑ์การตัดสินใจ:**
- ถ้าคำถามเป็นการค้นหาข้อเท็จจริงเดียว (single fact) → ตอบว่า `{{"needs_planning": false}}`
- ถ้าคำถามมีหลายส่วน (comparison, cross-doc, multi-aspect, list of items) → decompose เป็น 2-{max_subq} sub-questions

**ตัวอย่าง:**
- "ผู้รับจ้างชื่ออะไร" → single fact → `{{"needs_planning": false}}`
- "โครงการอะไรที่ปรากฏทั้งใน A และ B" → cross-doc → decompose
  ตอบ: `{{"needs_planning": true, "subqueries": ["โครงการใน A มีอะไร", "โครงการใน B มีอะไร"]}}`
- "เกณฑ์คะแนนแต่ละหมวดเป็นเท่าไหร่บ้าง" → multi-fact → decompose

**ตอบเป็น JSON เท่านั้น (ห้ามคำอธิบายอื่น):**
- Simple: `{{"needs_planning": false}}`
- Complex: `{{"needs_planning": true, "subqueries": ["...", "..."]}}`
"""


def _parse_plan(text: str) -> Dict[str, Any]:
    """Extract JSON from planner output, tolerant to prose/markdown."""
    m = re.search(r"\{[\s\S]*\}", text)
    if not m:
        return {"needs_planning": False}
    try:
        d = json.loads(m.group(0))
        if not isinstance(d, dict):
            return {"needs_planning": False}
        # Sanity: cap sub-queries
        subs = d.get("subqueries", []) or []
        if isinstance(subs, list):
            subs = [str(s).strip() for s in subs if s and isinstance(s, str)][:AGENTIC_MAX_SUBQ]
        return {
            "needs_planning": bool(d.get("needs_planning", False)) and bool(subs),
            "subqueries": subs,
        }
    except Exception:
        return {"needs_planning": False}


def plan_query(query: str) -> Dict[str, Any]:
    """Ask the planner LLM to decide if the query needs decomposition.
    Returns dict: {needs_planning: bool, subqueries: List[str]}
    Fail-open — any error returns {needs_planning: False}.
    """
    if not AGENTIC_ENABLED or not query or len(query) < AGENTIC_MIN_QUERY_LEN:
        return {"needs_planning": False, "subqueries": []}

    # Quick heuristic — skip planner LLM if no complexity hints
    if not _COMPLEX_HINTS.search(query):
        return {"needs_planning": False, "subqueries": []}

    try:
        import litellm
        from ingestion.config import CUSTOM_API_BASE, CUSTOM_API_KEY
    except Exception as e:
        logger.warning(f"[agentic] planner deps missing: {e}")
        return {"needs_planning": False, "subqueries": []}

    prompt = PLANNER_PROMPT.format(query=query, max_subq=AGENTIC_MAX_SUBQ)

    try:
        resp = litellm.completion(
            model=f"openai/{AGENTIC_MODEL}",
            messages=[{"role": "user", "content": prompt}],
            api_base=CUSTOM_API_BASE,
            api_key=CUSTOM_API_KEY,
            temperature=0.0,
            max_tokens=250,
            timeout=30,
        )
        text = (resp.choices[0].message.content or "").strip()
    except Exception as e:
        logger.warning(f"[agentic] planner call failed: {e}")
        return {"needs_planning": False, "subqueries": []}

    plan = _parse_plan(text)
    if plan.get("needs_planning"):
        logger.info(
            f"[agentic] planned {len(plan.get('subqueries', []))} subqueries · "
            f"orig='{query[:60]}' → {plan.get('subqueries')}"
        )
    return plan
