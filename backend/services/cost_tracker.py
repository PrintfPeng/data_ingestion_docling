"""
backend/services/cost_tracker.py

Persist per-call cost telemetry to a rolling daily JSONL log.

Each entry captures: timestamp, endpoint (llm/ocr/judge/rewriter/planner/grader),
provider (local/api/unknown), model, prompt/completion tokens, USD cost, and
optional context (doc_id, page). The log lives in `logs/costs/YYYY-MM-DD.jsonl`
inside the container.

Aggregation helpers roll today's file into totals for the UI meter, plus a
"session" window (any calls since backend startup — process-local, not per-user).

Fail-open: any I/O or arithmetic error is swallowed so tracking never blocks
the underlying request.
"""
from __future__ import annotations

import os
import json
import time
import logging
import threading
from pathlib import Path
from datetime import datetime, date, timedelta
from typing import Optional, Dict, Any, List

from .pricing import calculate_cost, _lookup_rate

logger = logging.getLogger(__name__)

# --- Config ---
COST_LOG_DIR = Path(os.getenv("COST_LOG_DIR", "logs/costs"))
COST_WARN_USD_PER_DAY = float(os.getenv("COST_WARN_USD_PER_DAY", "5.0"))
COST_TRACKING_ENABLED = os.getenv("COST_TRACKING_ENABLED", "true").lower() not in ("false", "0", "no")

_write_lock = threading.Lock()
_SESSION_START = time.time()  # process start — used for "session total" window


def _log_path(d: Optional[date] = None) -> Path:
    d = d or date.today()
    return COST_LOG_DIR / f"{d.isoformat()}.jsonl"


def log_call(
    endpoint: str,
    provider: str,
    model: str,
    prompt_tokens: int,
    completion_tokens: int,
    context: Optional[Dict[str, Any]] = None,
    user_id: Optional[int] = None,
) -> float:
    """Record one billable call. Returns the calculated USD cost (0 on failure).

    endpoint: "llm" (answer), "ocr", "judge", "rewriter", "planner", "grader"
    provider: "local" | "api" | "unknown"
    user_id: caller (0 = system / APP_API_KEY, positive = authed user)
    """
    if not COST_TRACKING_ENABLED:
        return 0.0
    try:
        cost = calculate_cost(model, prompt_tokens or 0, completion_tokens or 0)
        rate_in, rate_out = _lookup_rate(model)
        entry = {
            "ts": datetime.utcnow().isoformat() + "Z",
            "endpoint": endpoint,
            "provider": provider,
            "model": model,
            "prompt_tokens": int(prompt_tokens or 0),
            "completion_tokens": int(completion_tokens or 0),
            "rate_input_per_m": rate_in,
            "rate_output_per_m": rate_out,
            "cost_usd": round(cost, 8),
            "user_id": int(user_id) if user_id is not None else 0,
        }
        if context:
            entry["ctx"] = context

        p = _log_path()
        p.parent.mkdir(parents=True, exist_ok=True)
        line = json.dumps(entry, ensure_ascii=False)
        with _write_lock:
            with p.open("a", encoding="utf-8") as f:
                f.write(line + "\n")
        return cost
    except Exception as e:
        logger.warning(f"[cost_tracker] log_call failed: {e}")
        return 0.0


def log_from_response(
    endpoint: str,
    provider: str,
    model: str,
    response: Any,
    context: Optional[Dict[str, Any]] = None,
    user_id: Optional[int] = None,
) -> float:
    """Convenience wrapper that pulls token counts from a litellm/OpenAI-style
    response object (has .usage.prompt_tokens etc.).
    """
    try:
        usage = getattr(response, "usage", None)
        if usage is None and isinstance(response, dict):
            usage = response.get("usage")
        if usage is None:
            return log_call(endpoint, provider, model, 0, 0, context, user_id=user_id)
        prompt = getattr(usage, "prompt_tokens", None)
        completion = getattr(usage, "completion_tokens", None)
        if prompt is None and isinstance(usage, dict):
            prompt = usage.get("prompt_tokens", 0)
            completion = usage.get("completion_tokens", 0)
        return log_call(endpoint, provider, model, prompt or 0, completion or 0, context, user_id=user_id)
    except Exception as e:
        logger.warning(f"[cost_tracker] log_from_response failed: {e}")
        return 0.0


# ---------- Aggregation ----------

def _iter_entries(days: int = 1) -> List[Dict[str, Any]]:
    """Read the last N days of log files, oldest first."""
    entries: List[Dict[str, Any]] = []
    today = date.today()
    for i in range(days - 1, -1, -1):
        d = today - timedelta(days=i)
        p = _log_path(d)
        if not p.exists():
            continue
        try:
            with p.open("r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        entries.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue
        except Exception as e:
            logger.warning(f"[cost_tracker] read failed for {p}: {e}")
    return entries


def _filter_by_user(entries: List[Dict[str, Any]], user_id: Optional[int]) -> List[Dict[str, Any]]:
    """When user_id is set (and > 0), keep only that user's entries.
    None or 0 means 'all users' (admin view / global aggregate)."""
    if not user_id or user_id <= 0:
        return entries
    return [e for e in entries if int(e.get("user_id", 0) or 0) == int(user_id)]


def get_daily_total(days: int = 1, user_id: Optional[int] = None) -> Dict[str, Any]:
    """Sum today's cost + per-endpoint + per-provider breakdown.
    Pass user_id to scope the aggregate to one user; leave None for global.
    """
    entries = _filter_by_user(_iter_entries(days=days), user_id)
    total = 0.0
    by_endpoint: Dict[str, float] = {}
    by_provider: Dict[str, float] = {}
    by_model: Dict[str, float] = {}
    count = 0
    for e in entries:
        c = float(e.get("cost_usd") or 0.0)
        total += c
        count += 1
        by_endpoint[e.get("endpoint", "unknown")] = by_endpoint.get(e.get("endpoint", "unknown"), 0.0) + c
        by_provider[e.get("provider", "unknown")] = by_provider.get(e.get("provider", "unknown"), 0.0) + c
        m = e.get("model", "unknown")
        by_model[m] = by_model.get(m, 0.0) + c
    return {
        "total_usd": round(total, 6),
        "call_count": count,
        "days": days,
        "user_id": user_id or 0,  # 0 = global
        "warn_threshold_usd": COST_WARN_USD_PER_DAY,
        "warn_pct": round((total / COST_WARN_USD_PER_DAY * 100), 1) if COST_WARN_USD_PER_DAY > 0 else 0,
        "by_endpoint": {k: round(v, 6) for k, v in by_endpoint.items()},
        "by_provider": {k: round(v, 6) for k, v in by_provider.items()},
        "by_model": {k: round(v, 6) for k, v in by_model.items()},
    }


def get_session_total(user_id: Optional[int] = None) -> Dict[str, Any]:
    """Cost since backend process started (optionally scoped to a user)."""
    entries = _filter_by_user(_iter_entries(days=1), user_id)
    session_start_iso = datetime.utcfromtimestamp(_SESSION_START).isoformat() + "Z"
    filtered = [e for e in entries if e.get("ts", "") >= session_start_iso]
    total = sum(float(e.get("cost_usd") or 0.0) for e in filtered)
    return {
        "session_start_utc": session_start_iso,
        "total_usd": round(total, 6),
        "call_count": len(filtered),
    }


def get_recent_calls(limit: int = 50, user_id: Optional[int] = None) -> List[Dict[str, Any]]:
    """Return the most recent N calls, newest first (optionally scoped to a user)."""
    entries = _filter_by_user(_iter_entries(days=1), user_id)
    return list(reversed(entries))[:limit]


def get_per_user_totals(days: int = 1) -> Dict[str, Any]:
    """Aggregate today's cost per user_id. Used by the admin dashboard."""
    totals: Dict[int, Dict[str, float]] = {}
    for e in _iter_entries(days=days):
        uid = int(e.get("user_id", 0) or 0)
        c = float(e.get("cost_usd") or 0.0)
        if uid not in totals:
            totals[uid] = {"total_usd": 0.0, "call_count": 0}
        totals[uid]["total_usd"] += c
        totals[uid]["call_count"] += 1
    return {str(uid): {"total_usd": round(v["total_usd"], 6), "call_count": int(v["call_count"])}
            for uid, v in totals.items()}
