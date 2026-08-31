"""
backend/services/hybrid_search.py

Hybrid retrieval: BM25 (lexical) + vector search (dense) merged via
Reciprocal Rank Fusion (RRF).

Great for Thai proper nouns, dates in Thai numerals (๒๕๖๙), contract IDs
etc. — cases where dense vectors alone miss the surface form.
"""
from __future__ import annotations

import os
import logging
from typing import List, Optional, Dict, Any, Tuple

from langchain_core.documents import Document

from .vector_store import (
    search_similar,
    get_vector_store,
    sanitize_doc_id,
    CHROMA_DIR,
    COLLECTION_NAME,
)

logger = logging.getLogger(__name__)

# Env config
HYBRID_ENABLED = os.getenv("HYBRID_ENABLED", "true").lower() not in ("false", "0", "no")
RRF_K = int(os.getenv("RRF_K", "60"))  # RRF smoothing constant
BM25_TOP_N = int(os.getenv("BM25_TOP_N", "30"))
VECTOR_TOP_N = int(os.getenv("VECTOR_TOP_N", "30"))


# --- BM25 index cache ---
_bm25 = None
_bm25_docs: List[Document] = []
_bm25_load_error: Optional[Exception] = None


def _tokenize(text: str) -> List[str]:
    """Tokenize Thai + English. Uses pythainlp if available."""
    if not text:
        return []
    try:
        from pythainlp.tokenize import word_tokenize
        toks = word_tokenize(text, keep_whitespace=False)
        return [t.lower() for t in toks if t and not t.isspace()]
    except Exception:
        import re
        return re.findall(r"\w+", text.lower())


def _build_bm25() -> None:
    """Load all chunks from ChromaDB and build a BM25 index."""
    global _bm25, _bm25_docs, _bm25_load_error
    try:
        from rank_bm25 import BM25Okapi
    except ImportError as e:
        _bm25_load_error = e
        logger.warning(f"[hybrid_search] rank_bm25 not installed: {e}")
        return

    try:
        vectordb = get_vector_store(CHROMA_DIR, COLLECTION_NAME)
        # Pull all chunks — includes documents + metadatas
        raw = vectordb._collection.get(include=["documents", "metadatas"])
        contents = raw.get("documents", []) or []
        metadatas = raw.get("metadatas", []) or []
        ids = raw.get("ids", []) or []

        docs: List[Document] = []
        tokenized: List[List[str]] = []
        for text, md, cid in zip(contents, metadatas, ids):
            md = dict(md or {})
            md.setdefault("chunk_id", cid)
            doc = Document(page_content=text or "", metadata=md)
            docs.append(doc)
            tokenized.append(_tokenize(doc.page_content))

        if not tokenized:
            logger.warning("[hybrid_search] No chunks found — BM25 index empty")
            _bm25 = None
            _bm25_docs = []
            return

        _bm25 = BM25Okapi(tokenized)
        _bm25_docs = docs
        logger.info(f"[hybrid_search] Built BM25 index with {len(docs)} chunks")
    except Exception as e:
        _bm25_load_error = e
        logger.exception(f"[hybrid_search] Failed to build BM25 index: {e}")


def invalidate_bm25() -> None:
    """Clear BM25 cache — called after new chunks are indexed."""
    global _bm25, _bm25_docs, _bm25_load_error
    _bm25 = None
    _bm25_docs = []
    _bm25_load_error = None
    logger.info("[hybrid_search] BM25 cache invalidated")


def _bm25_search(query: str, k: int) -> List[Tuple[Document, float]]:
    """Return top-k documents by BM25 score."""
    global _bm25
    if _bm25 is None:
        _build_bm25()
    if _bm25 is None or not _bm25_docs:
        return []
    tokens = _tokenize(query)
    if not tokens:
        return []
    try:
        scores = _bm25.get_scores(tokens)
    except Exception as e:
        logger.warning(f"[hybrid_search] BM25 scoring failed: {e}")
        return []
    scored = sorted(
        enumerate(scores), key=lambda x: float(x[1]), reverse=True
    )[:k]
    return [(_bm25_docs[i], float(s)) for i, s in scored if s > 0]


def _filter_by_doc_ids(docs: List[Document], doc_ids: List[str]) -> List[Document]:
    wanted = set(sanitize_doc_id(d) for d in doc_ids if d)
    out = []
    for d in docs:
        did = str(d.metadata.get("doc_id", ""))
        if sanitize_doc_id(did) in wanted:
            out.append(d)
    return out


def _rrf_merge(
    vector_docs: List[Document],
    bm25_docs: List[Document],
    k_constant: int = RRF_K,
) -> List[Document]:
    """Reciprocal Rank Fusion: combine rankings by summing 1/(k+rank)."""
    scores: Dict[str, float] = {}
    doc_by_key: Dict[str, Document] = {}

    def key_of(d: Document) -> str:
        md = d.metadata or {}
        return str(md.get("chunk_id") or md.get("id") or d.page_content[:80])

    for rank, d in enumerate(vector_docs):
        key = key_of(d)
        scores[key] = scores.get(key, 0.0) + 1.0 / (k_constant + rank + 1)
        doc_by_key[key] = d
    for rank, d in enumerate(bm25_docs):
        key = key_of(d)
        scores[key] = scores.get(key, 0.0) + 1.0 / (k_constant + rank + 1)
        doc_by_key.setdefault(key, d)

    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return [doc_by_key[k] for k, _ in ranked]


def is_available() -> bool:
    if not HYBRID_ENABLED:
        return False
    try:
        from rank_bm25 import BM25Okapi  # noqa: F401
        return True
    except ImportError:
        return False


def hybrid_search_multi(
    queries: List[str],
    k: int = 30,
    doc_ids: Optional[List[str]] = None,
    persist_directory: str = CHROMA_DIR,
    collection_name: str = COLLECTION_NAME,
) -> List[Document]:
    """Run hybrid_search for each query variant, merge all results with RRF.
    Great with query_rewriter which produces 3-6 semantic/rule variants."""
    if not queries:
        return []
    if len(queries) == 1:
        return hybrid_search(queries[0], k, doc_ids, persist_directory, collection_name)

    all_rankings: List[List[Document]] = []
    for q in queries:
        try:
            res = hybrid_search(q, k=k, doc_ids=doc_ids,
                                persist_directory=persist_directory,
                                collection_name=collection_name)
            all_rankings.append(res)
        except Exception as e:
            logger.warning(f"[hybrid_search_multi] variant '{q[:40]}' failed: {e}")

    # RRF merge across all rankings
    scores: Dict[str, float] = {}
    doc_by_key: Dict[str, Document] = {}
    for ranking in all_rankings:
        for rank, d in enumerate(ranking):
            md = d.metadata or {}
            key = str(md.get("chunk_id") or md.get("id") or d.page_content[:80])
            scores[key] = scores.get(key, 0.0) + 1.0 / (RRF_K + rank + 1)
            doc_by_key.setdefault(key, d)

    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:k]
    merged = [doc_by_key[k] for k, _ in ranked]
    logger.info(
        f"[hybrid_search_multi] {len(queries)} variants · "
        f"union={len(doc_by_key)} unique · returning={len(merged)}"
    )
    return merged


def hybrid_search(
    query: str,
    k: int = 30,
    doc_ids: Optional[List[str]] = None,
    persist_directory: str = CHROMA_DIR,
    collection_name: str = COLLECTION_NAME,
) -> List[Document]:
    """Hybrid retrieval: run vector + BM25 in parallel, merge via RRF.
    Falls back to pure vector search if BM25 unavailable.
    """
    if not is_available():
        return search_similar(query=query, k=k, doc_ids=doc_ids,
                              persist_directory=persist_directory,
                              collection_name=collection_name)

    # Fetch more from each retriever than we'll return — RRF benefits from breadth
    vector_docs = search_similar(
        query=query, k=max(VECTOR_TOP_N, k), doc_ids=doc_ids,
        persist_directory=persist_directory, collection_name=collection_name,
    )
    bm25_pairs = _bm25_search(query, k=max(BM25_TOP_N, k))
    bm25_docs = [d for d, _ in bm25_pairs]

    # doc_id filtering — BM25 didn't know about the filter
    if doc_ids:
        bm25_docs = _filter_by_doc_ids(bm25_docs, doc_ids)

    merged = _rrf_merge(vector_docs, bm25_docs)
    top = merged[:k]

    logger.info(
        f"[hybrid_search] vector={len(vector_docs)} · bm25={len(bm25_docs)} "
        f"· merged={len(merged)} · returning={len(top)}"
    )
    return top
