from __future__ import annotations

from typing import List, Optional, Literal, Dict, Any
from pathlib import Path
import shutil
import subprocess
import sys
import os
import re
import time
import json
from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Header, Depends
from fastapi.responses import RedirectResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

# -----------------------------------------------------------
# API Key Authentication
# -----------------------------------------------------------
# If APP_API_KEY env var is not set (or empty), auth is DISABLED
# and all endpoints work without an Authorization header.
# If set, clients must send: Authorization: Bearer <key>
APP_API_KEY = (os.getenv("APP_API_KEY") or "").strip()


def verify_api_key(authorization: Optional[str] = Header(None)) -> None:
    """FastAPI dependency: validate the Bearer token in Authorization header.
    Skips validation entirely when APP_API_KEY is not configured (dev mode).
    """
    if not APP_API_KEY:
        return  # auth disabled
    if not authorization or not authorization.lower().startswith("bearer "):
        raise HTTPException(
            status_code=401,
            detail="Missing Authorization header. Set API key in Settings.",
        )
    token = authorization.split(" ", 1)[1].strip()
    if token != APP_API_KEY:
        raise HTTPException(status_code=403, detail="Invalid API key")

# Internal services
from .services.logger import append_log, read_logs
from .services.rag import answer_question, answer_question_stream
from .services.vector_store import reset_vector_store_cache

# Config Paths
INGESTED_DIR = Path("ingested")
CHROMA_DB_DIR = Path("chroma_db")
UPLOAD_DIR = Path("uploads")

# -----------------------------------------------------------
# FastAPI App Setup
# -----------------------------------------------------------
app = FastAPI(
    title="AI Data Ingestion Backend",
    description="Backend for DB, Embeddings, RAG, API, and Evaluation",
    version="0.2.2 (Multi-Doc Final)",
)

# 1. Mount Frontend
frontend_path = Path(__file__).resolve().parents[1] / "frontend"
app.mount("/app", StaticFiles(directory=str(frontend_path), html=True), name="frontend")

# 2. Mount Ingested Data
INGESTED_DIR.mkdir(parents=True, exist_ok=True)
app.mount("/ingested", StaticFiles(directory=str(INGESTED_DIR)), name="ingested")

# 3. Upload Directory
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)


# -----------------------------------------------------------
# Helper: ID Normalization
# -----------------------------------------------------------
def _normalize_id(raw_id: str) -> str:
    if not raw_id:
        return "unknown_doc"
    s = raw_id.strip().lower()
    s = re.sub(r"\s+", "_", s)
    s = re.sub(r"[^a-z0-9_\-\u0E00-\u0E7F]", "", s)
    return s


# -----------------------------------------------------------
# Health Check
# -----------------------------------------------------------
@app.get("/health")
async def health_check():
    return {
        "status": "ok",
        "service": "backend",
        "mode": "multi_doc",
        "features": ["hybrid_ingestion", "ocr", "rag"],
    }


# -----------------------------------------------------------
# /ask (RAG + Hybrid Rendering)
# -----------------------------------------------------------
class AskRequest(BaseModel):
    query: str
    doc_ids: Optional[List[str]] = None
    top_k: int = 5
    mode: Literal["auto", "text", "table", "both"] = "auto"
    # [NEW] รับประวัติการแชท (List of dicts: [{"role": "user", "content": "..."}, ...])
    history: List[Dict[str, str]] = []
    # Phase 2: per-query LLM mode selector (local vs cloud API)
    llm_mode: Literal["auto", "local", "api"] = "auto"

class AskResponse(BaseModel):
    answer: str
    sources: List[dict]
    intent: str
    mode: str
    tables: List[Dict[str, Any]] = []
    # Phase 2 — surface the LLM used so the UI can display it
    llm_mode: Optional[str] = None
    llm_provider: Optional[str] = None
    llm_model: Optional[str] = None
    # Phase 4 — per-request cost estimate (USD)
    cost_estimate_usd: Optional[float] = None

@app.post("/ask", response_model=AskResponse, dependencies=[Depends(verify_api_key)])
async def ask(req: AskRequest):
    # 1. Normalize IDs
    sanitized_doc_ids = None
    if req.doc_ids:
        sanitized_doc_ids = [_normalize_id(did) for did in req.doc_ids if did]

    # 2. Call RAG Service
    result = await answer_question(
        query=req.query,
        doc_ids=sanitized_doc_ids,
        top_k=req.top_k,
        mode=req.mode,
        history=req.history, # [NEW] ส่ง history ไปให้ rag service ด้วย
        llm_mode=req.llm_mode,
    )

    # Post-Processing: Convert [SHOW_TABLE] tags
    answer_text = result.get("answer", "")
    sources = result.get("sources", [])
    
    table_tags = re.findall(r"\[SHOW_TABLE:CAT=(.*?)\]", answer_text)

    for category_key in table_tags:
        clean_cat = category_key.strip()
        replacement_html = ""

        for src in sources:
            metadata = src.get("metadata", src)
            
            is_table_source = src.get("source") == "table" or metadata.get("source") == "table"
            is_image_source = src.get("source") == "image" or metadata.get("source") == "image"
            
            if is_table_source or is_image_source:
                src_cat = metadata.get("category", "")
                if (src_cat == clean_cat) or (clean_cat == ""):
                    
                    # Case A: Image
                    image_path = metadata.get("image_path") or metadata.get("extra", {}).get("image_path")
                    if image_path:
                        doc_id = metadata.get("doc_id")
                        full_img_url = f"/ingested/{doc_id}/{image_path}"
                        replacement_html = (
                            f"<div class='my-4 p-2 border rounded bg-slate-50 text-center'>"
                            f"<p class='text-xs text-slate-500 mb-1'>Original Form (Complex Layout)</p>"
                            f"<img src='{full_img_url}' alt='Table Image' "
                            f"class='max-w-full h-auto rounded shadow-sm mx-auto border' />"
                            f"</div>"
                        )
                        break

                    # Case B: HTML
                    html_content = metadata.get("html_content") or metadata.get("extra", {}).get("html_content")
                    if html_content:
                        replacement_html = f"<br><div class='table-responsive answer-tables-content'>{html_content}</div><br>"
                        break
        
        tag_str = f"[SHOW_TABLE:CAT={category_key}]"
        if replacement_html:
            answer_text = answer_text.replace(tag_str, replacement_html)
        else:
            answer_text = answer_text.replace(tag_str, "")

    result["answer"] = answer_text

    # Logging
    try:
        append_log({
            "query": req.query, "doc_ids": req.doc_ids,
            "answer": result.get("answer"), "intent": result.get("intent")
        })
    except Exception as e:
        print(f"[LOG_ERROR] {e!r}")

    result["tables"] = result.get("tables", [])
    return AskResponse(**result)


# -----------------------------------------------------------
# /history
# -----------------------------------------------------------
# -----------------------------------------------------------
# /ask/stream — SSE streaming variant of /ask
# -----------------------------------------------------------
@app.post("/ask/stream", dependencies=[Depends(verify_api_key)])
async def ask_stream(req: AskRequest):
    """Stream the answer token-by-token via Server-Sent Events.
    Emits the retrieved sources first, then tokens, then a done event.
    """
    sanitized_doc_ids = None
    if req.doc_ids:
        sanitized_doc_ids = [_normalize_id(d) for d in req.doc_ids if d]

    async def event_stream():
        try:
            async for event_name, payload in answer_question_stream(
                query=req.query,
                doc_ids=sanitized_doc_ids,
                top_k=req.top_k,
                mode=req.mode,
                history=req.history,
                llm_mode=req.llm_mode,
            ):
                yield f"event: {event_name}\ndata: {json.dumps(payload, ensure_ascii=False)}\n\n"
        except Exception as e:
            yield f"event: error\ndata: {json.dumps({'message': str(e)}, ensure_ascii=False)}\n\n"

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",  # tell nginx not to buffer
            "Connection": "keep-alive",
        },
    )


class HistoryItem(BaseModel):
    ts: str
    query: str
    answer: str
    doc_ids: Optional[List[str]] = None
    intent: Optional[str] = None
    mode: Optional[str] = None

@app.get("/history", response_model=List[HistoryItem], dependencies=[Depends(verify_api_key)])
def get_history(limit: int = 50):
    logs = read_logs(limit=limit)
    items = []
    for e in logs:
        items.append(HistoryItem(
            ts=e.get("ts", ""), query=e.get("query", ""), answer=e.get("answer", ""),
            doc_ids=e.get("doc_ids"), intent=e.get("intent"), mode=e.get("mode")
        ))
    return items


# -----------------------------------------------------------
# /upload (Multi-Document Mode)
# -----------------------------------------------------------
_VALID_OCR_MODES = {"auto", "local", "api"}


@app.post("/upload", dependencies=[Depends(verify_api_key)])
async def upload_pdf(
    file: UploadFile = File(...),
    doc_id: str = Form(...),
    doc_type: str = Form(""),
    use_ocr: bool = Form(True),
    ocr_mode: str = Form("auto"),
):
    # 0. Defaults
    if not doc_type.strip(): doc_type = "generic_doc"

    # 1. Validation
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="รองรับเฉพาะไฟล์ PDF เท่านั้น")
    if not doc_id.strip():
        raise HTTPException(status_code=400, detail="ต้องระบุ doc_id")

    ocr_mode = (ocr_mode or "auto").lower().strip()
    if ocr_mode not in _VALID_OCR_MODES:
        raise HTTPException(status_code=400, detail=f"ocr_mode ต้องเป็น auto/local/api (ได้: {ocr_mode})")
    if ocr_mode == "api" and not (os.getenv("VISION_API_KEY") or "").strip():
        raise HTTPException(status_code=400, detail="ocr_mode='api' ต้องตั้ง VISION_API_KEY บน server ก่อน")

    safe_doc_id = _normalize_id(doc_id)
    print(f"[UPLOAD] Received doc_id='{doc_id}' -> normalized='{safe_doc_id}'")

    # 2. Ensure Folders Exist (No Cleanup = Multi-Doc)
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    INGESTED_DIR.mkdir(parents=True, exist_ok=True)

    # 3. Save File
    dest_path = UPLOAD_DIR / f"{safe_doc_id}.pdf"
    try:
        with dest_path.open("wb") as f:
            shutil.copyfileobj(file.file, f)
    finally:
        file.file.close()

    # 4. Run Ingestion Pipeline (with cost snapshot)
    try:
        from backend.services.cost_tracker import get_daily_total as _cost_snap
        _cost_before = _cost_snap(days=1).get("total_usd", 0.0)
    except Exception:
        _cost_before = 0.0

    try:
        print(f"[UPLOAD] 🛑 Releasing DB lock before ingestion...")
        reset_vector_store_cache()

        script_name = "scripts.run_ingestion" if use_ocr else "scripts.run_all"
        cmd = [
            sys.executable, "-m", script_name,
            str(dest_path),
            "--doc-id", safe_doc_id,
            "--doc-type", doc_type,
            "--output-root", str(INGESTED_DIR)
        ]
        if script_name == "scripts.run_ingestion":
            if not use_ocr:
                cmd.append("--no-ocr")
            cmd.extend(["--ocr-mode", ocr_mode])
            
        print(f"[UPLOAD] Running pipeline: {' '.join(cmd)}")
        subprocess.run(cmd, check=True)

    except subprocess.CalledProcessError as e:
        raise HTTPException(status_code=500, detail=f"Ingestion pipeline failed: {e}")

    # 5. Re-index Vector DB (Append Mode)
    reset_vector_store_cache()
    try:
        # สคริปต์นี้จะสแกน ingested folder ทั้งหมด (เก่า+ใหม่) แล้ว Index รวมกัน
        cmd = [sys.executable, "-m", "scripts.ingest_doc"]
        print(f"[UPLOAD] Re-indexing (All Docs): {' '.join(cmd)}")
        subprocess.run(cmd, check=True)
        
        print("[UPLOAD] ⏳ Waiting for DB lock release (3s)...")
        time.sleep(3)
    except subprocess.CalledProcessError as e:
        raise HTTPException(status_code=500, detail=f"Re-index failed: {e}")

    # Clear Cache
    reset_vector_store_cache()

    # Cost delta from the ingest run (best-effort — 0 for local OCR)
    try:
        _cost_after = _cost_snap(days=1).get("total_usd", 0.0)
        _cost_upload = max(0.0, _cost_after - _cost_before)
    except Exception:
        _cost_upload = 0.0

    return {
        "ok": True,
        "doc_id": safe_doc_id,
        "original_doc_id": doc_id,
        "doc_type": doc_type,
        "ocr_mode": ocr_mode,
        "cost_estimate_usd": round(_cost_upload, 6),
        "message": "File uploaded and ingested successfully (Append Mode).",
        "pipeline": "hybrid_ingestion",
    }


# -----------------------------------------------------------
# /documents (List Documents)
# -----------------------------------------------------------
# -----------------------------------------------------------
# Phase 4 — cost telemetry endpoints
# -----------------------------------------------------------
@app.get("/stats/cost", dependencies=[Depends(verify_api_key)])
def stats_cost(days: int = 1):
    """Aggregate cost for the last N days plus process-lifetime session total."""
    from backend.services.cost_tracker import get_daily_total, get_session_total
    daily = get_daily_total(days=max(1, min(days, 30)))
    session = get_session_total()
    return {
        "daily": daily,
        "session": session,
    }


@app.get("/stats/cost/recent", dependencies=[Depends(verify_api_key)])
def stats_cost_recent(limit: int = 50):
    from backend.services.cost_tracker import get_recent_calls
    return {"calls": get_recent_calls(limit=max(1, min(limit, 500)))}


@app.get("/documents", dependencies=[Depends(verify_api_key)])
def list_documents():
    docs = []
    if INGESTED_DIR.exists():
        for item in INGESTED_DIR.iterdir():
            if item.is_dir():
                docs.append({
                    "id": item.name,
                    "name": item.name 
                })
    docs.sort(key=lambda x: x["name"])
    return {"documents": docs}


# -----------------------------------------------------------
# Root Redirect
# -----------------------------------------------------------
@app.get("/")
def root():
    return RedirectResponse(url="/app/index.html")