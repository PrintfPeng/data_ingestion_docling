"""
backend/services/reranker.py

Cross-encoder reranker (BAAI/bge-reranker-v2-m3) — takes top-N candidates
from vector search, scores each (query, chunk) pair, returns top-K reordered.

Loads lazily on first call, caches singleton for the process.
"""
from __future__ import annotations

import os
import logging
from typing import List, Optional, Tuple, Any

logger = logging.getLogger(__name__)

DEFAULT_MODEL = os.getenv("RERANKER_MODEL", "BAAI/bge-reranker-v2-m3")


def _auto_device() -> str:
    """Prefer GPU if available; env RERANKER_DEVICE overrides (cpu/cuda/cuda:0)."""
    explicit = os.getenv("RERANKER_DEVICE", "").strip()
    if explicit:
        return explicit
    try:
        import torch
        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"


DEFAULT_DEVICE = _auto_device()

_reranker = None
_load_error: Optional[Exception] = None


def _get_reranker():
    """Lazy-load and cache the CrossEncoder."""
    global _reranker, _load_error
    if _reranker is not None:
        return _reranker
    if _load_error is not None:
        return None
    try:
        from sentence_transformers import CrossEncoder
        logger.info(f"[reranker] Loading {DEFAULT_MODEL} on {DEFAULT_DEVICE}...")
        _reranker = CrossEncoder(DEFAULT_MODEL, device=DEFAULT_DEVICE, max_length=512)
        logger.info(f"[reranker] Loaded {DEFAULT_MODEL}")
        return _reranker
    except Exception as e:
        logger.exception(f"[reranker] Failed to load: {e}")
        _load_error = e
        return None


def is_available() -> bool:
    """Return True if reranking is possible (env allows + model loadable)."""
    if os.getenv("RERANK_ENABLED", "true").lower() in ("false", "0", "no"):
        return False
    return _get_reranker() is not None


def rerank(query: str, docs: List[Any], top_k: int = 5) -> List[Any]:
    """Rerank a list of langchain Document objects by relevance to `query`.
    Falls back to original order (truncated to top_k) if reranker unavailable.

    Args:
        query: user question
        docs: list with `.page_content` attribute (langchain Documents)
        top_k: keep top-K after reranking

    Returns:
        Reordered docs, len == min(len(docs), top_k)
    """
    if not docs:
        return docs
    if not is_available() or len(docs) <= 1:
        return docs[:top_k]

    model = _get_reranker()
    pairs: List[Tuple[str, str]] = [(query, d.page_content) for d in docs]
    try:
        scores = model.predict(pairs, batch_size=16, show_progress_bar=False)
    except Exception as e:
        logger.warning(f"[reranker] predict failed: {e} — returning original order")
        return docs[:top_k]

    scored = sorted(zip(scores, docs), key=lambda x: float(x[0]), reverse=True)

    # Stash score in metadata so downstream filters can act on it
    top = []
    for score, d in scored[:top_k]:
        try:
            d.metadata["rerank_score"] = float(score)
        except Exception:
            pass
        top.append(d)

    if top:
        logger.info(
            f"[reranker] {len(docs)} → {len(top)} · "
            f"top score={float(scored[0][0]):.3f}"
        )
    return top
