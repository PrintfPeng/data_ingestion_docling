from __future__ import annotations

from typing import List, Optional, Literal, Dict, Any
from pathlib import Path
import shutil
import subprocess
import sys
import os
import re
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import RedirectResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

# Internal services
from .services.logger import append_log, read_logs
from .services.rag import answer_question
from .services.vector_store import reset_vector_store_cache


# -----------------------------------------------------------
# FastAPI App Setup
# -----------------------------------------------------------

app = FastAPI(
    title="AI Data Ingestion Backend",
    description="Backend for DB, Embeddings, RAG, API, and Evaluation",
    version="0.2.0 (Hybrid Supported)",
)

# 1. Mount Frontend (Static Assets)
frontend_path = Path(__file__).resolve().parents[1] / "frontend"
app.mount(
    "/app",
    StaticFiles(directory=str(frontend_path), html=True),
    name="frontend",
)

# 2. [CRITICAL] Mount Ingested Data (Images/Tables)
# ต้องแน่ใจว่า folder นี้มีอยู่จริง เพื่อกัน error 500 ตอน start app
ingested_path = Path("ingested")
ingested_path.mkdir(parents=True, exist_ok=True)

app.mount(
    "/ingested", 
    StaticFiles(directory=str(ingested_path)), 
    name="ingested"
)

# 3. Upload Directory
UPLOAD_DIR = Path("uploads")
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)


# -----------------------------------------------------------
# Helper: ID Normalization
# -----------------------------------------------------------
def _normalize_id(raw_id: str) -> str:
    """
    ทำให้ ID เป็นมาตรฐานเดียวกัน (Lowercase, No Spaces, Safe Chars)
    """
    if not raw_id:
        return "unknown_doc"
    
    # 1. Lowercase & Strip
    s = raw_id.strip().lower()
    
    # 2. Replace spaces with underscores
    s = re.sub(r"\s+", "_", s)
    
    # 3. Remove weird chars (Allow Thai chars \u0E00-\u0E7F)
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
        "version": "0.2.0",
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


class AskResponse(BaseModel):
    answer: str
    sources: List[dict]
    intent: str
    mode: str
    tables: List[Dict[str, Any]] = []


@app.post("/ask", response_model=AskResponse)
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
    )

    # =================================================================
    # [HYBRID FIX] Post-Processing: Convert [SHOW_TABLE] tags
    # รองรับทั้ง HTML Table และ Image Table
    # =================================================================
    answer_text = result.get("answer", "")
    sources = result.get("sources", [])
    
    # regex หา tag [SHOW_TABLE:CAT=...]
    table_tags = re.findall(r"\[SHOW_TABLE:CAT=(.*?)\]", answer_text)

    for category_key in table_tags:
        clean_cat = category_key.strip()
        replacement_html = ""

        # Scan sources to find matching table/image
        for src in sources:
            metadata = src.get("metadata", src) # Fallback
            
            # Check source type
            is_table_source = src.get("source") == "table" or metadata.get("source") == "table"
            is_image_source = src.get("source") == "image" or metadata.get("source") == "image"
            
            if is_table_source or is_image_source:
                src_cat = metadata.get("category", "")
                
                # Match Category (Empty cat matches first table found)
                if (src_cat == clean_cat) or (clean_cat == ""):
                    
                    # Case A: Complex Table -> Image
                    image_path = metadata.get("image_path") or metadata.get("extra", {}).get("image_path")
                    if image_path:
                        # Construct Image URL (assuming /ingested/ is mounted)
                        # image_path มักจะเป็น "images/filename.png"
                        # ต้องเอา doc_id มาประกอบ path: /ingested/{doc_id}/{image_path}
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

                    # Case B: Simple Table -> HTML
                    html_content = metadata.get("html_content") or metadata.get("extra", {}).get("html_content")
                    if html_content:
                        replacement_html = f"<br><div class='table-responsive answer-tables-content'>{html_content}</div><br>"
                        break
        
        # Replace Tag
        tag_str = f"[SHOW_TABLE:CAT={category_key}]"
        if replacement_html:
            answer_text = answer_text.replace(tag_str, replacement_html)
        else:
            # Not found -> Remove tag
            answer_text = answer_text.replace(tag_str, "")

    # Update result
    result["answer"] = answer_text
    # =================================================================

    # 3. Logging
    try:
        append_log({
            "query": req.query,
            "doc_ids": req.doc_ids,
            "sanitized_ids": sanitized_doc_ids,
            "top_k": req.top_k,
            "mode": req.mode,
            "answer": result.get("answer"),
            "intent": result.get("intent"),
            "sources": result.get("sources"),
        })
    except Exception as e:
        print(f"[LOG_ERROR] {e!r}")

    # 4. Return
    result["tables"] = result.get("tables", [])
    return AskResponse(**result)


# -----------------------------------------------------------
# /history
# -----------------------------------------------------------

class HistoryItem(BaseModel):
    ts: str
    query: str
    answer: str
    doc_ids: Optional[List[str]] = None
    intent: Optional[str] = None
    mode: Optional[str] = None


@app.get("/history", response_model=List[HistoryItem])
def get_history(limit: int = 50):
    logs = read_logs(limit=limit)
    items: List[HistoryItem] = []

    for e in logs:
        items.append(HistoryItem(
            ts=e.get("ts", ""),
            query=e.get("query", ""),
            answer=e.get("answer", ""),
            doc_ids=e.get("doc_ids"),
            intent=e.get("intent"),
            mode=e.get("mode"),
        ))

    return items


# -----------------------------------------------------------
# /upload
# -----------------------------------------------------------

@app.post("/upload")
async def upload_pdf(
    file: UploadFile = File(...),
    doc_id: str = Form(...),
    doc_type: str = Form(""),
    use_ocr: bool = Form(True),
):
    """
    1) Normalize doc_id
    2) Save PDF
    3) Run Ingestion (Hybrid Support via run_ingestion.py)
    4) Re-index (ingest_doc.py)
    """

    # 0. Defaults
    if not doc_type or not doc_type.strip():
        doc_type = "generic_doc"

    # 1. Validation
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="รองรับเฉพาะไฟล์ PDF เท่านั้น")

    if not doc_id.strip():
        raise HTTPException(status_code=400, detail="ต้องระบุ doc_id")

    # 2. Normalize ID
    safe_doc_id = _normalize_id(doc_id)
    print(f"[UPLOAD] Received doc_id='{doc_id}' -> normalized='{safe_doc_id}'")

    # 3. Save File
    dest_path = UPLOAD_DIR / f"{safe_doc_id}.pdf"
    try:
        with dest_path.open("wb") as f:
            shutil.copyfileobj(file.file, f)
    finally:
        file.file.close()

    # 4. Run Ingestion Pipeline
    try:
        # Determine script to run
        # NOTE: เราแนะนำให้ใช้ scripts.run_ingestion เสมอเพราะรองรับ Hybrid Ingestion
        script_name = "scripts.run_ingestion" if use_ocr else "scripts.run_all"
        
        cmd = [
            sys.executable,
            "-m",
            script_name,
            str(dest_path),
            "--doc-id", safe_doc_id,
            "--doc-type", doc_type,
        ]
        
        # scripts.run_ingestion รองรับ --no-ocr ถ้า user ไม่ต้องการ
        if script_name == "scripts.run_ingestion" and not use_ocr:
            cmd.append("--no-ocr")
            
        print(f"[UPLOAD] Running pipeline: {' '.join(cmd)}")
        subprocess.run(cmd, check=True)

    except subprocess.CalledProcessError as e:
        raise HTTPException(
            status_code=500,
            detail=f"Ingestion pipeline failed: {e}",
        ) from e

    # 5. Re-index Vector DB (ingest_doc.py handles Hybrid Metadata)
    try:
        cmd = [sys.executable, "-m", "backend.scripts.ingest_doc"]
        print(f"[UPLOAD] Re-indexing: {' '.join(cmd)}")
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        raise HTTPException(
            status_code=500,
            detail=f"Re-index failed: {e}",
        ) from e

    # Clear Cache
    reset_vector_store_cache()

    return {
        "ok": True,
        "doc_id": safe_doc_id,
        "original_doc_id": doc_id,
        "doc_type": doc_type,
        "pipeline": "hybrid_ingestion",
    }


@app.get("/documents")
def list_documents():
    """
    List all documents in 'ingested' folder
    """
    ingested_root = Path("ingested")
    docs = []
    if ingested_root.exists():
        for item in ingested_root.iterdir():
            if item.is_dir():
                docs.append({
                    "id": item.name,
                    "name": item.name # Can be enhanced if we store display name
                })
    
    docs.sort(key=lambda x: x["name"])
    return {"documents": docs}


# -----------------------------------------------------------
# Root Redirect
# -----------------------------------------------------------

@app.get("/")
def root():
    return RedirectResponse(url="/app/index.html")