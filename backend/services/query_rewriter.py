"""
backend/services/query_rewriter.py

Rewrites a user query into multiple equivalent variants to improve retrieval
recall. Combines:

1. **Rule-based normalization** — deterministic, zero-latency:
   - Thai numerals ↔ Arabic digits (๕ ↔ 5)
   - Number words → digits (ห้า → 5, ๕)
   - Buddhist year (พ.ศ.) ↔ Gregorian (ค.ศ.)

2. **LLM semantic variants** — optional (env-toggled), uses a fast LLM
   (default qwen2.5:7b via Ollama) to generate 1-2 paraphrases.
   Cached in-process to avoid re-calling on repeat queries.

Returned list always begins with the ORIGINAL query, then unique variants.
"""
from __future__ import annotations

import os
import re
import logging
from functools import lru_cache
from typing import List, Optional

logger = logging.getLogger(__name__)


# ---------------- Config ----------------
LLM_REWRITE_ENABLED = os.getenv("QUERY_REWRITE_LLM", "true").lower() not in ("false", "0", "no")
LLM_REWRITE_MODEL = os.getenv("QUERY_REWRITE_MODEL", "qwen2.5:7b")
LLM_REWRITE_MAX_VARIANTS = int(os.getenv("QUERY_REWRITE_MAX", "2"))
RULE_REWRITE_ENABLED = os.getenv("QUERY_REWRITE_RULES", "true").lower() not in ("false", "0", "no")


# ---------------- Rule-based normalization ----------------
_THAI_TO_ARABIC = str.maketrans("๐๑๒๓๔๕๖๗๘๙", "0123456789")
_ARABIC_TO_THAI = str.maketrans("0123456789", "๐๑๒๓๔๕๖๗๘๙")

# ตัวเลขไทยแบบคำ → digit (ครอบคลุม 0-20 + สิบ, ร้อย, พัน)
_NUMBER_WORDS = {
    "ศูนย์": "0", "หนึ่ง": "1", "สอง": "2", "สาม": "3", "สี่": "4",
    "ห้า": "5", "หก": "6", "เจ็ด": "7", "แปด": "8", "เก้า": "9",
    "สิบ": "10", "สิบเอ็ด": "11", "สิบสอง": "12", "สิบสาม": "13",
    "สิบสี่": "14", "สิบห้า": "15", "สิบหก": "16", "สิบเจ็ด": "17",
    "สิบแปด": "18", "สิบเก้า": "19", "ยี่สิบ": "20",
    "ร้อย": "100", "พัน": "1000", "หมื่น": "10000",
}
_NUMBER_WORDS_SORTED = sorted(_NUMBER_WORDS.items(), key=lambda x: -len(x[0]))


def _to_arabic_digits(s: str) -> str:
    return s.translate(_THAI_TO_ARABIC)

def _to_thai_digits(s: str) -> str:
    return s.translate(_ARABIC_TO_THAI)

def _replace_number_words(s: str) -> str:
    """Replace Thai number-words with Arabic digits.
    Uses pythainlp tokenizer so that "ห้า" is only replaced when it is a
    standalone token — never inside "ห้าง"/"ห้าม" etc.
    Falls back to raw substring replace if pythainlp unavailable (best-effort).
    """
    try:
        from pythainlp.tokenize import word_tokenize
        tokens = word_tokenize(s, keep_whitespace=True)
        return "".join(_NUMBER_WORDS.get(t, t) for t in tokens)
    except Exception:
        # Fallback (may produce false positives on words containing number-words)
        out = s
        for word, digit in _NUMBER_WORDS_SORTED:
            out = out.replace(word, digit)
        return out

def _shift_year(s: str, delta: int) -> str:
    """Shift 4-digit years appearing near พ.ศ./ค.ศ. or standalone (2500-2700)."""
    def repl(m: re.Match) -> str:
        y = int(m.group(0))
        if 2400 <= y <= 2700 and delta:  # only shift plausible BE years
            return str(y + delta)
        return m.group(0)
    return re.sub(r"\d{4}", repl, s)


def rule_variants(query: str) -> List[str]:
    """Deterministic query variants derived from numeric/date rules."""
    variants = set()

    variants.add(query)
    variants.add(_to_arabic_digits(query))
    variants.add(_to_thai_digits(query))

    # Replace number words → digits, then re-generate both scripts
    words_replaced = _replace_number_words(query)
    if words_replaced != query:
        variants.add(words_replaced)
        variants.add(_to_thai_digits(words_replaced))

    # Buddhist year ↔ Gregorian year: 2569 ↔ 2026
    ary = _to_arabic_digits(query)
    variants.add(_shift_year(ary, -543))   # BE → CE
    variants.add(_shift_year(ary, +543))   # CE → BE

    # Return unique, preserving original first
    out = [query]
    for v in variants:
        if v != query and v not in out:
            out.append(v)
    return out


# ---------------- LLM semantic variants ----------------
@lru_cache(maxsize=200)
def _llm_variants(query: str, n: int, model: str) -> tuple:
    """Ask a fast LLM for n paraphrases. Returns tuple (for lru_cache)."""
    try:
        from ingestion.config import CUSTOM_API_BASE, CUSTOM_API_KEY
        from openai import OpenAI
    except Exception as e:
        logger.warning(f"[query_rewriter] LLM deps missing: {e}")
        return ()

    prompt = f"""คุณคือระบบเขียนคำถามใหม่สำหรับ RAG search
ให้เขียนคำถามใหม่ที่มีความหมายเหมือนคำถามเดิม จำนวน {n} เวอร์ชัน
ใช้ synonyms หรือรูปประโยคที่ต่างจากเดิมเพื่อเพิ่มโอกาสค้นเจอ

คำถามเดิม: {query}

ตอบเป็น JSON list เท่านั้น เช่น: ["คำถามใหม่ 1", "คำถามใหม่ 2"]
ห้ามใส่คำอธิบายอื่น"""
    try:
        client = OpenAI(api_key=CUSTOM_API_KEY, base_url=CUSTOM_API_BASE)
        resp = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
            max_tokens=300,
        )
        text = (resp.choices[0].message.content or "").strip()
        import json
        m = re.search(r"\[.*?\]", text, re.DOTALL)
        if not m:
            return ()
        arr = json.loads(m.group(0))
        return tuple(str(x).strip() for x in arr if isinstance(x, str) and x.strip())
    except Exception as e:
        logger.warning(f"[query_rewriter] LLM rewrite failed: {e}")
        return ()


def llm_variants(query: str, n: int = LLM_REWRITE_MAX_VARIANTS) -> List[str]:
    if not LLM_REWRITE_ENABLED or n <= 0:
        return []
    return list(_llm_variants(query, n, LLM_REWRITE_MODEL))


# ---------------- Query intent classification ----------------
# Mirrors the intent scoring in backend/services/chunking.py so query-side
# classification matches how chunks were tagged during ingestion.
_QI_TROUBLESHOOT = re.compile(r"(?:แก้ปัญหา|error|fail|not\s*working|เสีย|ซ่อม|troubleshoot)", re.IGNORECASE)
_QI_SAFETY = re.compile(r"(?:ความปลอดภัย|warning|danger|ระวัง|ห้าม|อันตราย)", re.IGNORECASE)
_QI_INSTALL = re.compile(r"(?:วิธี|ขั้นตอน|how\s*to|install|setup|การติดตั้ง|วิธีการ)", re.IGNORECASE)
_QI_IDENTITY = re.compile(r"(?:ผู้|ชื่อ|ลงนาม|อนุมัติ|who|name|signature|ใคร)", re.IGNORECASE)
_QI_FINANCE = re.compile(r"(?:ราคา|ค่าใช้จ่าย|เงิน|บาท|cost|price|งวด|จ่าย|ค่าจ้าง)", re.IGNORECASE)
_QI_REF = re.compile(r"(?:ความหมาย|คือ|definition|spec|สเปค|คุณลักษณะ|เกี่ยวกับอะไร)", re.IGNORECASE)


def classify_query_intent(query: str) -> Optional[str]:
    """Return the primary intent of a query, or None if no signal found.
    Uses the same regex scoring as chunk-side intent tagging so filters match."""
    if not query:
        return None
    scores = {}
    if _QI_TROUBLESHOOT.search(query): scores["troubleshooting"] = 3
    if _QI_SAFETY.search(query):       scores["safety"] = 3
    if _QI_INSTALL.search(query):      scores["installation"] = 2
    if _QI_IDENTITY.search(query):     scores["identity"] = 2
    if _QI_FINANCE.search(query):      scores["financial"] = 2
    if _QI_REF.search(query):          scores["reference"] = 1
    if not scores:
        return None
    return max(scores.items(), key=lambda x: x[1])[0]


# ---------------- Public API ----------------
def rewrite_query(query: str, use_llm: Optional[bool] = None) -> List[str]:
    """Return unique query variants — original first, then rule-based, then LLM.
    Empty query returns single-element list with the (empty) original."""
    if not query or not query.strip():
        return [query]

    variants: List[str] = [query]

    if RULE_REWRITE_ENABLED:
        for v in rule_variants(query):
            if v not in variants:
                variants.append(v)

    do_llm = LLM_REWRITE_ENABLED if use_llm is None else use_llm
    if do_llm:
        for v in llm_variants(query):
            if v not in variants:
                variants.append(v)

    logger.info(f"[query_rewriter] '{query[:60]}' → {len(variants)} variants")
    return variants
