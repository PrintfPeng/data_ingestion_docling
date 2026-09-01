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


# Internal services
from .services.logger import append_log, read_logs
from .services.rag import answer_question, answer_question_stream
from .services.vector_store import reset_vector_store_cache
from .services import users as users_svc
from .services import sessions as sessions_svc
from .services.db import init_db


# The "system" user represents anyone using the legacy shared APP_API_KEY
# — backward compat for eval scripts + CLI tools. Treated as admin.
_SYSTEM_USER: Dict[str, Any] = {
    "id": 0,
    "username": "system",
    "email": None,
    "is_admin": True,
    "auth_kind": "app_key",
}


def _extract_bearer(authorization: Optional[str]) -> Optional[str]:
    if not authorization:
        return None
    parts = authorization.split(" ", 1)
    if len(parts) != 2 or parts[0].lower() != "bearer":
        return None
    return parts[1].strip() or None


def verify_api_key(authorization: Optional[str] = Header(None)) -> None:
    """Legacy dependency — keep for endpoints that still use it.
    Accepts APP_API_KEY OR a valid session token. Skips validation entirely
    when APP_API_KEY is empty AND no users are registered (initial dev mode).
    """
    token = _extract_bearer(authorization)
    if not APP_API_KEY and not token:
        return  # first-boot / dev mode
    if token and APP_API_KEY and token == APP_API_KEY:
        return
    if token:
        sess = sessions_svc.verify_token(token)
        if sess:
            return
    raise HTTPException(status_code=401, detail="Missing or invalid Authorization")


def current_user(authorization: Optional[str] = Header(None)) -> Dict[str, Any]:
    """FastAPI dependency: return the current user dict.
    Accepts either APP_API_KEY (→ system user) or a session token.
    """
    token = _extract_bearer(authorization)
    if not token:
        raise HTTPException(status_code=401, detail="Missing Authorization header")
    if APP_API_KEY and token == APP_API_KEY:
        return dict(_SYSTEM_USER)
    sess = sessions_svc.verify_token(token)
    if not sess:
        raise HTTPException(status_code=401, detail="Invalid or expired session")
    user = users_svc.get_user_by_id(sess["user_id"])
    if not user or user.get("disabled_at"):
        raise HTTPException(status_code=403, detail="User disabled")
    user["auth_kind"] = "session"
    return user


def require_admin(user: Dict[str, Any] = Depends(current_user)) -> Dict[str, Any]:
    if not user.get("is_admin"):
        raise HTTPException(status_code=403, detail="Admin only")
    return user


# Ensure the users DB is ready at process start
try:
    init_db()
except Exception as e:
    print(f"[main] init_db warning: {e}")

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

# -----------------------------------------------------------
# Phase 5.2 — Auth endpoints (login/logout/me)
# -----------------------------------------------------------
class LoginRequest(BaseModel):
    username: str
    password: str


class LoginResponse(BaseModel):
    token: str
    user_id: int
    username: str
    is_admin: bool
    expires_at: str


@app.post("/auth/login", response_model=LoginResponse)
def auth_login(req: LoginRequest):
    user = users_svc.verify_credentials(req.username, req.password)
    if not user:
        raise HTTPException(status_code=401, detail="Invalid username or password")
    session = sessions_svc.create_session(user["id"])
    return {
        "token": session["token"],
        "user_id": user["id"],
        "username": user["username"],
        "is_admin": bool(user["is_admin"]),
        "expires_at": session["expires_at"],
    }


@app.post("/auth/logout")
def auth_logout(authorization: Optional[str] = Header(None)):
    token = _extract_bearer(authorization)
    if token and (not APP_API_KEY or token != APP_API_KEY):
        sessions_svc.revoke_token(token)
    return {"ok": True}


@app.get("/auth/me")
def auth_me(user: Dict[str, Any] = Depends(current_user)):
    return {
        "id": user.get("id"),
        "username": user.get("username"),
        "email": user.get("email"),
        "is_admin": bool(user.get("is_admin")),
        "auth_kind": user.get("auth_kind", "session"),
    }


# -----------------------------------------------------------
# Phase 5.4 — Per-user settings (OpenRouter key vault)
# -----------------------------------------------------------
class UserSettingsResponse(BaseModel):
    default_preset: str
    has_openrouter_key: bool


class SetOpenrouterKeyRequest(BaseModel):
    key: str  # empty string clears the key


class SetPresetRequest(BaseModel):
    preset: Literal["air_gapped", "hybrid", "cloud_premium"]


def _user_id_or_400(user: Dict[str, Any]) -> int:
    uid = user.get("id")
    if not uid or uid <= 0:
        raise HTTPException(status_code=400, detail="/me endpoints require a logged-in user (not system)")
    return int(uid)


@app.get("/me/settings", response_model=UserSettingsResponse)
def get_me_settings(user: Dict[str, Any] = Depends(current_user)):
    uid = _user_id_or_400(user)
    s = users_svc.get_settings(uid)
    return {
        "default_preset": s.get("default_preset", "hybrid"),
        "has_openrouter_key": bool(s.get("openrouter_key_encrypted")),
    }


@app.put("/me/settings/openrouter_key")
def set_me_openrouter_key(
    req: SetOpenrouterKeyRequest,
    user: Dict[str, Any] = Depends(current_user),
):
    uid = _user_id_or_400(user)
    from backend.services import keyvault
    if req.key.strip():
        blob = keyvault.encrypt(req.key.strip())
        users_svc.set_openrouter_key(uid, blob)
        return {"ok": True, "has_openrouter_key": True}
    users_svc.set_openrouter_key(uid, None)
    return {"ok": True, "has_openrouter_key": False}


@app.put("/me/settings/default_preset")
def set_me_default_preset(
    req: SetPresetRequest,
    user: Dict[str, Any] = Depends(current_user),
):
    uid = _user_id_or_400(user)
    users_svc.set_default_preset(uid, req.preset)
    return {"ok": True, "default_preset": req.preset}


@app.post("/ask", response_model=AskResponse)
async def ask(req: AskRequest, user: Dict[str, Any] = Depends(current_user)):
    # 1. Normalize IDs
    sanitized_doc_ids = None
    if req.doc_ids:
        sanitized_doc_ids = [_normalize_id(did) for did in req.doc_ids if did]

    # 2. Call RAG Service — pass user_id so cloud calls use their key
    result = await answer_question(
        query=req.query,
        doc_ids=sanitized_doc_ids,
        top_k=req.top_k,
        mode=req.mode,
        history=req.history, # [NEW] ส่ง history ไปให้ rag service ด้วย
        llm_mode=req.llm_mode,
        user_id=user.get("id") or None,
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
@app.post("/ask/stream")
async def ask_stream(req: AskRequest, user: Dict[str, Any] = Depends(current_user)):
    """Stream the answer token-by-token via Server-Sent Events.
    Emits the retrieved sources first, then tokens, then a done event.
    """
    sanitized_doc_ids = None
    if req.doc_ids:
        sanitized_doc_ids = [_normalize_id(d) for d in req.doc_ids if d]

    caller_id = user.get("id") or None

    async def event_stream():
        try:
            async for event_name, payload in answer_question_stream(
                query=req.query,
                doc_ids=sanitized_doc_ids,
                top_k=req.top_k,
                mode=req.mode,
                history=req.history,
                llm_mode=req.llm_mode,
                user_id=caller_id,
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


@app.post("/upload")
async def upload_pdf(
    file: UploadFile = File(...),
    doc_id: str = Form(...),
    doc_type: str = Form(""),
    use_ocr: bool = Form(True),
    ocr_mode: str = Form("auto"),
    user: Dict[str, Any] = Depends(current_user),
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
            
        # Phase 5.4: pass user's OpenRouter key to OCR subprocess via env
        # (VISION_API_KEY_OVERRIDE takes priority over VISION_API_KEY inside the subprocess).
        # Phase 5.5: also pass OCR_USER_ID so cost_tracker attributes OCR cost to the caller.
        subprocess_env = os.environ.copy()
        uid = user.get("id") or 0
        subprocess_env["OCR_USER_ID"] = str(uid)
        if uid > 0:
            try:
                from backend.services import keyvault
                settings = users_svc.get_settings(uid)
                blob = settings.get("openrouter_key_encrypted")
                if blob:
                    user_key = keyvault.decrypt(blob)
                    if user_key:
                        subprocess_env["VISION_API_KEY_OVERRIDE"] = user_key
                        print(f"[UPLOAD] using user_id={uid} OpenRouter key for OCR")
            except Exception as e:
                print(f"[UPLOAD] user key resolution failed: {e}")

        print(f"[UPLOAD] Running pipeline: {' '.join(cmd)}")
        subprocess.run(cmd, check=True, env=subprocess_env)

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
# Phase 4 + 5.5 — cost telemetry endpoints (scoped by user)
# -----------------------------------------------------------
def _effective_scope_user(user: Dict[str, Any], scope: Optional[str]) -> Optional[int]:
    """Return the user_id to filter cost by. `scope=all` is admin-only."""
    if scope == "all":
        if not user.get("is_admin"):
            raise HTTPException(status_code=403, detail="scope=all requires admin")
        return None  # global aggregate
    uid = user.get("id") or 0
    # System user (APP_API_KEY) → sees global aggregate too
    if uid <= 0:
        return None
    return int(uid)


@app.get("/stats/cost")
def stats_cost(
    days: int = 1,
    scope: Optional[str] = None,
    user: Dict[str, Any] = Depends(current_user),
):
    """Aggregate cost for the last N days + session total.
    Non-admin users see only their own; admin can pass ?scope=all for global.
    """
    from backend.services.cost_tracker import get_daily_total, get_session_total
    uid = _effective_scope_user(user, scope)
    daily = get_daily_total(days=max(1, min(days, 30)), user_id=uid)
    session = get_session_total(user_id=uid)
    return {
        "daily": daily,
        "session": session,
        "scope": "all" if uid is None else f"user:{uid}",
        "is_admin": bool(user.get("is_admin")),
    }


@app.get("/stats/cost/recent")
def stats_cost_recent(
    limit: int = 50,
    scope: Optional[str] = None,
    user: Dict[str, Any] = Depends(current_user),
):
    from backend.services.cost_tracker import get_recent_calls
    uid = _effective_scope_user(user, scope)
    return {"calls": get_recent_calls(limit=max(1, min(limit, 500)), user_id=uid)}


# -----------------------------------------------------------
# Phase 5.6 — Admin dashboard endpoints
# -----------------------------------------------------------
class CreateUserRequest(BaseModel):
    username: str
    password: str
    email: Optional[str] = None
    is_admin: bool = False


class ResetPasswordRequest(BaseModel):
    new_password: str


class SetDisabledRequest(BaseModel):
    disabled: bool


@app.get("/admin/users")
def admin_list_users(user: Dict[str, Any] = Depends(require_admin)):
    """List all users + their daily spend (today's total_usd)."""
    from backend.services.cost_tracker import get_per_user_totals
    users = users_svc.list_users()
    totals = get_per_user_totals(days=1)  # {"user_id": {"total_usd": ..., "call_count": ...}}
    for u in users:
        stat = totals.get(str(u["id"])) or {"total_usd": 0.0, "call_count": 0}
        u["daily_cost_usd"] = stat["total_usd"]
        u["daily_call_count"] = stat["call_count"]
    return {"users": users}


@app.post("/admin/users")
def admin_create_user(req: CreateUserRequest, user: Dict[str, Any] = Depends(require_admin)):
    try:
        u = users_svc.create_user(
            username=req.username,
            password=req.password,
            email=req.email,
            is_admin=req.is_admin,
        )
        return {"ok": True, "user": u}
    except users_svc.UserError as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.put("/admin/users/{user_id}/password")
def admin_reset_password(user_id: int, req: ResetPasswordRequest, user: Dict[str, Any] = Depends(require_admin)):
    target = users_svc.get_user_by_id(user_id)
    if not target:
        raise HTTPException(status_code=404, detail="user not found")
    try:
        users_svc.set_password(user_id, req.new_password)
    except users_svc.UserError as e:
        raise HTTPException(status_code=400, detail=str(e))
    # Revoke all sessions so old logins can't keep working with the old password
    sessions_svc.revoke_all_for_user(user_id)
    return {"ok": True, "sessions_revoked": True}


@app.put("/admin/users/{user_id}/disabled")
def admin_set_disabled(user_id: int, req: SetDisabledRequest, user: Dict[str, Any] = Depends(require_admin)):
    target = users_svc.get_user_by_id(user_id)
    if not target:
        raise HTTPException(status_code=404, detail="user not found")
    if user_id == user["id"] and req.disabled:
        raise HTTPException(status_code=400, detail="cannot disable your own admin account")
    users_svc.set_disabled(user_id, req.disabled)
    if req.disabled:
        sessions_svc.revoke_all_for_user(user_id)
    return {"ok": True, "disabled": req.disabled}


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