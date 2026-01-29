from __future__ import annotations

from typing import List, Optional, Literal, Dict, Any
from pathlib import Path
import shutil
import subprocess
import sys
import os
import re  # [NEW] Import re
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import RedirectResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from .services.logger import append_log, read_logs
from .services.rag import answer_question

from .services.vector_store import reset_vector_store_cache


# -----------------------------------------------------------
# FastAPI app & Static frontend
# -----------------------------------------------------------

app = FastAPI(
    title="AI Data Ingestion Backend",
    description="Backend for DB, Embeddings, RAG, API, and Evaluation",
    version="0.1.0",
)

# เสิร์ฟไฟล์ frontend (index.html + assets) ที่ /app/
frontend_path = Path(__file__).resolve().parents[1] / "frontend"
app.mount(
    "/app",
    StaticFiles(directory=str(frontend_path), html=True),
    name="frontend",
)

app.mount("/ingested", StaticFiles(directory="ingested"), name="ingested")
# โฟลเดอร์สำหรับอัปโหลดไฟล์ PDF ใหม่
UPLOAD_DIR = Path("uploads")
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)


# -----------------------------------------------------------
# Helper: ID Normalization (CRITICAL FIX)
# -----------------------------------------------------------
def _normalize_id(raw_id: str) -> str:
    """
    ทำให้ ID เป็นมาตรฐานเดียวกันทั้งระบบ (Backend/Frontend/DB)
    - เปลี่ยนเป็นตัวพิมพ์เล็ก
    - แทนที่ช่องว่างด้วย _
    - ลบอักขระพิเศษ (เก็บ a-z, 0-9, _, - และภาษาไทยไว้)
    """
    if not raw_id:
        return "unknown_doc"
    
    # 1. Lowercase & Strip
    s = raw_id.strip().lower()
    
    # 2. Replace spaces with underscores
    s = re.sub(r"\s+", "_", s)
    
    # 3. Remove weird chars
    # [CHANGE] แก้บรรทัดนี้: เพิ่ม \u0E00-\u0E7F เพื่อรองรับภาษาไทย
    s = re.sub(r"[^a-z0-9_\-\u0E00-\u0E7F]", "", s)
    
    return s


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
    # [FIX] เพิ่ม field tables เพื่อรองรับ Frontend schema (แม้เราจะ render html ใน text ก็ตาม)
    tables: List[Dict[str, Any]] = []


@app.post("/ask", response_model=AskResponse)
async def ask(req: AskRequest):
    
    # [FIX] Normalize doc_ids before sending to RAG
    # เพื่อให้ตรงกับตอน Upload (แก้ปัญหา ID Mismatch)
    sanitized_doc_ids = None
    if req.doc_ids:
        sanitized_doc_ids = [_normalize_id(did) for did in req.doc_ids if did]

    # 1) เรียก RAG ตอบคำถาม
    result = await answer_question(
        query=req.query,
        doc_ids=sanitized_doc_ids, # Use sanitized IDs
        top_k=req.top_k,
        mode=req.mode,
    )

    # =================================================================
    # [FIX] Post-Processing: แปลง Tag [SHOW_TABLE] เป็น HTML
    # =================================================================
    answer_text = result.get("answer", "")
    sources = result.get("sources", [])
    
    # หา Tag ทั้งหมดในคำตอบ
    table_tags = re.findall(r"\[SHOW_TABLE:CAT=(.*?)\]", answer_text)

    for category_key in table_tags:
        clean_cat = category_key.strip()
        found_html = ""

        # วนหา HTML Content จาก Sources ที่ RAG คืนมา
        for src in sources:
            # ตรวจสอบว่าเป็น Table Source หรือไม่
            # (ต้องดู structure ของ src ว่าเก็บ html_content ไว้ตรงไหน โดยปกติจะอยู่ใน metadata หรือ root keys)
            # กรณีนี้เราจะพยายามหาจากหลายๆ ที่เพื่อความชัวร์
            metadata = src.get("metadata", src) # Fallback to src itself if metadata key missing
            
            is_table = src.get("source") == "table" or metadata.get("source") == "table"
            
            if is_table:
                # ดึง Category และ HTML
                src_cat = metadata.get("category", "")
                src_html = metadata.get("html_content") or metadata.get("extra", {}).get("html_content")

                # Match Category (ถ้า Tag ไม่ระบุ Category ให้ถือว่าเอาตารางแรกที่เจอ)
                if (src_cat == clean_cat) or (clean_cat == ""):
                    if src_html:
                        found_html = src_html
                        break
        
        # แทนที่ Tag ในคำตอบ
        tag_str = f"[SHOW_TABLE:CAT={category_key}]"
        if found_html:
            #
            # แทรก HTML ลงไปใน text (Frontend จะ render ให้เองเพราะมี DOMPurify)
            replacement = f"<br><div class='table-responsive'>{found_html}</div><br>"
            answer_text = answer_text.replace(tag_str, replacement)
        else:
            # ถ้าหาตารางไม่เจอ ให้ลบ Tag ออกเพื่อความสะอาด
            answer_text = answer_text.replace(tag_str, "")

    # อัปเดตคำตอบกลับเข้าไปใน result
    result["answer"] = answer_text
    # =================================================================

    # 2) เขียน log ลงไฟล์ (กันไม่ให้ทำ API พังถ้า log มีปัญหา)
    try:
        append_log(
            {
                "query": req.query,
                "doc_ids": req.doc_ids, # Log original IDs for debugging
                "sanitized_ids": sanitized_doc_ids,
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
    # ใส่ tables=[] เพื่อกัน error validation (เพราะ model เราเพิ่ม field นี้)
    result["tables"] = result.get("tables", [])
    
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

    # [FIX] Normalize ID ทันทีที่รับมา
    # นี่คือจุดสำคัญที่สุด: เปลี่ยน "Operation Manual Sharp" -> "operation_manual_sharp"
    safe_doc_id = _normalize_id(doc_id)
    
    print(f"[UPLOAD] Received doc_id='{doc_id}' -> normalized='{safe_doc_id}'")

    # 2) เซฟไฟล์ลง uploads/ โดยใช้ safe_doc_id
    dest_path = UPLOAD_DIR / f"{safe_doc_id}.pdf"
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
                safe_doc_id, # Use sanitized ID
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
                safe_doc_id, # Use sanitized ID
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
    reset_vector_store_cache()
    return {
        "ok": True,
        "doc_id": safe_doc_id, # Return normalized ID
        "original_doc_id": doc_id,
        "doc_type": doc_type,
        "use_ocr": use_ocr,
    }


@app.get("/documents")
def list_documents():
    """
    คืนรายชื่อเอกสารทั้งหมดที่มีในโฟลเดอร์ ingested/
    เพื่อให้ Frontend เอาไปสร้าง Dropdown
    """
    ingested_root = Path("ingested")
    docs = []
    if ingested_root.exists():
        # Scan folder names inside 'ingested'
        for item in ingested_root.iterdir():
            if item.is_dir():
                # [FIX] Return both ID and Display Name
                # ID = folder name (which is normalized)
                # Name = folder name (can be improved if we stored mapping, but this is consistent)
                docs.append({
                    "id": item.name,
                    "name": item.name
                })
    
    # Sort for consistency
    docs.sort(key=lambda x: x["name"])
    return {"documents": docs}

# -----------------------------------------------------------
# Root redirect -> /app
# -----------------------------------------------------------

@app.get("/")
def root():
    # redirect ไปหน้า frontend หลัก
    return RedirectResponse(url="/app/index.html")