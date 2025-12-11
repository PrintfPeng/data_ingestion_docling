from __future__ import annotations

from typing import List, Optional, Literal
from pathlib import Path
import shutil
import subprocess
import sys

from fastapi.middleware.cors import CORSMiddleware

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import RedirectResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from .services.logger import append_log, read_logs
from .services.rag import answer_question


# -----------------------------------------------------------
# FastAPI app & Static frontend
# -----------------------------------------------------------

app = FastAPI(
    title="AI Data Ingestion Backend",
    description="Backend for DB, Embeddings, RAG, API, and Evaluation",
    version="0.1.0",
)
# --- เพิ่มส่วนนี้ลงไปหลังจากสร้างตัวแปร app ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # อนุญาตทุก Origin (หรือใส่ ["http://localhost:3000"])
    allow_credentials=True,
    allow_methods=["*"],  # อนุญาตทุก Method (GET, POST, etc.)
    allow_headers=["*"],  # อนุญาตทุก Header
)
# ------------------------------------------------
# เสิร์ฟไฟล์ frontend (index.html + assets) ที่ /app/
frontend_path = Path(__file__).resolve().parents[1] / "frontend"
app.mount(
    "/app",
    StaticFiles(directory=str(frontend_path), html=True),
    name="frontend",
)

# โฟลเดอร์สำหรับอัปโหลดไฟล์ PDF ใหม่
UPLOAD_DIR = Path("uploads")
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)


# -----------------------------------------------------------
# Health check
# -----------------------------------------------------------

@app.get("/health")
async def health_check():
    return {
        "status": "ok",
        "service": "backend",
        "version": "0.1.0",
    }


# -----------------------------------------------------------
# /ask (RAG + Intent + Logging)
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


@app.post("/ask", response_model=AskResponse)
async def ask(req: AskRequest):
    # 1) เรียก RAG ตอบคำถาม
    result = await answer_question(
        query=req.query,
        doc_ids=req.doc_ids,
        top_k=req.top_k,
        mode=req.mode,
    )

    # 2) เขียน log ลงไฟล์ (กันไม่ให้ทำ API พังถ้า log มีปัญหา)
    try:
        append_log(
            {
                "query": req.query,
                "doc_ids": req.doc_ids,
                "top_k": req.top_k,
                "mode": req.mode,
                "answer": result.get("answer"),
                "intent": result.get("intent"),
                "sources": result.get("sources"),
            }
        )
    except Exception as e:  # noqa: BLE001
        print(f"[LOG_ERROR] {e!r}")

    # 3) คืนค่าเป็น AskResponse (ตอบตรงตาม schema)
    return AskResponse(**result)


# -----------------------------------------------------------
# /history  (อ่าน log ย้อนหลัง)
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
    """
    ดึง history Q&A ย้อนหลังใหม่สุดไม่เกิน limit รายการ
    """
    logs = read_logs(limit=limit)
    items: List[HistoryItem] = []

    for e in logs:
        items.append(
            HistoryItem(
                ts=e.get("ts", ""),
                query=e.get("query", ""),
                answer=e.get("answer", ""),
                doc_ids=e.get("doc_ids"),
                intent=e.get("intent"),
                mode=e.get("mode"),
            )
        )

    return items


# -----------------------------------------------------------
# /upload  (อัปโหลด PDF -> ingestion -> ingest_doc)
# -----------------------------------------------------------

@app.post("/upload")
async def upload_pdf(
    file: UploadFile = File(...),
    doc_id: str = Form(...),
    # เดิม: doc_type: str = Form("bank_statement")
    # ใหม่: ให้ปล่อยว่างได้ ถ้าไม่ส่งมาจะ default = "generic_doc"
    doc_type: str = Form(""),
    # เปิดให้เลือกได้ว่าจะใช้ OCR pipeline หรือไม่
    # - True  = ใช้ scripts.run_ingestion (มี OCR ช่วยอ่าน)
    # - False = ใช้ scripts.run_all แบบเดิมที่ Peng เขียนไว้
    use_ocr: bool = Form(True),
):
    """
    1) รับไฟล์ PDF จากผู้ใช้
    2) เซฟลง uploads/<doc_id>.pdf
    3) ถ้า use_ocr=True  -> ใช้ scripts.run_ingestion (OCR + ingestion)
       ถ้า use_ocr=False -> ใช้ scripts.run_all (behavior เดิมทั้งชุด)
    4) เรียก backend.scripts.ingest_doc เพื่อ re-index vector DB
    """

    # 0) normalize doc_type ให้มีค่าเสมอ
    if not doc_type or not doc_type.strip():
        doc_type = "generic_doc"

    # 1) ตรวจไฟล์
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="รองรับเฉพาะไฟล์ PDF เท่านั้น")

    if not doc_id.strip():
        raise HTTPException(status_code=400, detail="ต้องระบุ doc_id")

    # 2) เซฟไฟล์ลง uploads/
    dest_path = UPLOAD_DIR / f"{doc_id}.pdf"
    try:
        with dest_path.open("wb") as f:
            shutil.copyfileobj(file.file, f)
    finally:
        file.file.close()

    # 3) เรียก ingestion pipeline
    try:
        if use_ocr:
            # ใหม่: pipeline ที่มี OCR ช่วยอ่าน ก่อนทำ text.json / table.json / image.json
            cmd = [
                sys.executable,
                "-m",
                "scripts.run_ingestion",
                str(dest_path),
                "--doc-id",
                doc_id,
                "--doc-type",
                doc_type,
            ]
            print(f"[UPLOAD] run with OCR: {' '.join(cmd)}")
            subprocess.run(cmd, check=True)
        else:
            # เดิม: ใช้ run_all ของ Peng ทั้งชุด (ไม่มี OCR)
            cmd = [
                sys.executable,
                "-m",
                "scripts.run_all",
                str(dest_path),
                "--doc-id",
                doc_id,
                "--doc-type",
                doc_type,
            ]
            print(f"[UPLOAD] run legacy pipeline (no OCR): {' '.join(cmd)}")
            subprocess.run(cmd, check=True)

    except subprocess.CalledProcessError as e:  # noqa: PERF203
        raise HTTPException(
            status_code=500,
            detail=f"ingestion pipeline error: {e}",
        ) from e

    # 4) re-index vector DB (backend.scripts.ingest_doc จะ scan โฟลเดอร์ ingested)
    try:
        cmd = [sys.executable, "-m", "backend.scripts.ingest_doc"]
        print(f"[UPLOAD] re-index vector DB: {' '.join(cmd)}")
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:  # noqa: PERF203
        raise HTTPException(
            status_code=500,
            detail=f"re-index error (ingest_doc): {e}",
        ) from e

    return {
        "ok": True,
        "doc_id": doc_id,
        "doc_type": doc_type,
        "use_ocr": use_ocr,
    }


# -----------------------------------------------------------
# Root redirect -> /app
# -----------------------------------------------------------

@app.get("/")
def root():
    # redirect ไปหน้า frontend หลัก
    return RedirectResponse(url="/app/")
