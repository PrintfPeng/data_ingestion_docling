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

app = FastAPI(
    title="AI Data Ingestion Backend",
    description="Backend for DB, Embeddings, RAG, API",
    version="0.1.0",
)

# Config CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# =================================================================
# 1. Mount Frontend & Images
# =================================================================
current_dir = Path(__file__).resolve().parent  # backend/
root_dir = current_dir.parent                  # root/
frontend_path = root_dir / "frontend"

app.mount("/app", StaticFiles(directory=str(frontend_path), html=True), name="frontend")

os.makedirs("ingested", exist_ok=True)
app.mount("/ingested", StaticFiles(directory="ingested"), name="ingested")

@app.get("/")
def root():
    return RedirectResponse(url="/app/")


# =================================================================
# 2. Schema สำหรับ /ask (ปรับให้ยืดหยุ่นรับ Frontend ได้แน่นอน)
# =================================================================
class AskRequest(BaseModel):
    query: str
    doc_ids: Optional[List[str]] = None
    top_k: Optional[int] = 5
    mode: Optional[str] = "auto" # ใช้ str ธรรมดาแทน Literal เพื่อลดปัญหา Error 400

@app.post("/ask")
async def ask(req: AskRequest):
    # ตรวจสอบ query
    if not req.query or not req.query.strip():
        raise HTTPException(status_code=400, detail="Missing query text")

    try:
        # [FIX] เรียกใช้ answer_question แบบ Synchronous (ไม่ใช้ await)
        # และส่งเฉพาะ query เพราะ rag.py ตัวใหม่รับแค่ตัวเดียว
        result = answer_question(req.query)
        
        # เติมข้อมูลที่ขาด (เพื่อให้ Frontend ไม่ error)
        if "intent" not in result: result["intent"] = "qa"
        if "mode" not in result: result["mode"] = req.mode
        
        return result

    except Exception as e:
        print(f"[ERROR] Ask failed: {e}")
        return {
            "answer": f"เกิดข้อผิดพลาด: {str(e)}",
            "sources": [],
            "intent": "error"
        }


# =================================================================
# 3. Upload Endpoint
# =================================================================
@app.post("/upload")
async def upload_document(
    file: UploadFile = File(...),
    doc_id: str = Form(...),
    doc_type: str = Form("generic_doc"), 
    use_ocr: bool = Form(True) # รับค่านี้ไว้แต่ไม่ใช้ก็ได้ กัน error
):
    upload_dir = "uploads"
    os.makedirs(upload_dir, exist_ok=True)
    
    file_path = os.path.join(upload_dir, file.filename)
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
        
    print(f"[UPLOAD] Starting ingestion for: {doc_id}")
    
    try:
        run_ingestion(
            pdf_path=file_path,
            doc_id=doc_id,
            doc_type=doc_type
        )
        return {"status": "success", "doc_id": doc_id, "message": "Ingestion complete"}
    except Exception as e:
        print(f"Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/documents")
def list_documents():
    ingested_root = Path("ingested")
    docs = []
    if ingested_root.exists():
        for item in ingested_root.iterdir():
            if item.is_dir():
                docs.append(item.name)
    docs.sort()
    return {"documents": docs}