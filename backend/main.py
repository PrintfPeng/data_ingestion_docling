from __future__ import annotations
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import RedirectResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from typing import List, Optional, Any
import shutil
import os
from pathlib import Path

# Import services
from .services.rag import answer_question
from .scripts.ingest_doc import run_ingestion

app = FastAPI(title="AI Data Ingestion Backend", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 1. Mount Frontend
current_dir = Path(__file__).resolve().parent
root_dir = current_dir.parent
frontend_path = root_dir / "frontend"
app.mount("/app", StaticFiles(directory=str(frontend_path), html=True), name="frontend")

# --- [FIX] กำหนด Path ตรงนี้ให้ชัดเจน ---
INGESTED_ROOT = Path(r"D:\DATA_INGES\ingested")
os.makedirs(INGESTED_ROOT, exist_ok=True)

# [FIX] ใช้ str(INGESTED_ROOT) เพื่อให้ Mount ไปที่ D:\... จริงๆ
app.mount("/ingested", StaticFiles(directory=str(INGESTED_ROOT)), name="ingested")


@app.get("/")
def root():
    return RedirectResponse(url="/app/")

class AskRequest(BaseModel):
    query: str
    doc_ids: Optional[List[str]] = None
    top_k: Optional[int] = 5
    mode: Optional[str] = "auto"

@app.post("/ask")
async def ask(req: AskRequest):
    if not req.query.strip(): raise HTTPException(status_code=400, detail="Missing query")
    try:
        result = answer_question(req.query)
        if "intent" not in result: result["intent"] = "qa"
        if "mode" not in result: result["mode"] = req.mode
        return result
    except Exception as e:
        print(f"[ERROR] Ask failed: {e}")
        return {"answer": f"Error: {str(e)}", "sources": [], "intent": "error"}

@app.post("/upload")
async def upload_document(
    file: UploadFile = File(...),
    doc_id: str = Form(...),
    doc_type: str = Form("generic_doc"), 
    use_ocr: bool = Form(True)
):
    upload_dir = "uploads"
    os.makedirs(upload_dir, exist_ok=True)
    file_path = os.path.join(upload_dir, file.filename)
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    
    print(f"[UPLOAD] Processing: {doc_id}")
    try:
        run_ingestion(
            pdf_path=file_path,
            doc_id=doc_id,
            doc_type=doc_type,
            output_root=INGESTED_ROOT 
        )
        return {"status": "success", "doc_id": doc_id, "message": "Ingestion complete"}
    except Exception as e:
        print(f"Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/documents")
def list_documents():
    docs = []
    if INGESTED_ROOT.exists():
        for item in INGESTED_ROOT.iterdir():
            if item.is_dir(): docs.append(item.name)
    docs.sort()
    return {"documents": docs}