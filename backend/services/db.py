"""
backend/services/db.py

SQLite connection + schema init for the multi-tenant user system (Phase 5).

Tables:
- users             — id, username, email, password_hash, is_admin, created_at, disabled_at
- user_settings     — user_id, default_preset, openrouter_key_encrypted, updated_at
- sessions          — token, user_id, created_at, expires_at, revoked_at
- (Phase 5.5) cost_events already uses JSONL — user_id will be added to each log entry

DB lives at DATA_DIR/users.db (default: backend/data/users.db inside the container).
Uses stdlib sqlite3 with WAL mode for basic concurrency.
"""
from __future__ import annotations

import os
import sqlite3
import threading
import logging
from pathlib import Path
from contextlib import contextmanager
from typing import Iterator

logger = logging.getLogger(__name__)

DATA_DIR = Path(os.getenv("APP_DATA_DIR", "backend/data")).resolve()
DB_PATH = DATA_DIR / "users.db"

_init_lock = threading.Lock()
_initialized = False


SCHEMA = """
CREATE TABLE IF NOT EXISTS users (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    username        TEXT NOT NULL UNIQUE COLLATE NOCASE,
    email           TEXT,
    password_hash   TEXT NOT NULL,
    is_admin        INTEGER NOT NULL DEFAULT 0,
    created_at      TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    disabled_at     TEXT
);

CREATE INDEX IF NOT EXISTS idx_users_username ON users(username);

CREATE TABLE IF NOT EXISTS user_settings (
    user_id                     INTEGER PRIMARY KEY,
    default_preset              TEXT NOT NULL DEFAULT 'hybrid',
    openrouter_key_encrypted    BLOB,
    updated_at                  TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY(user_id) REFERENCES users(id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS sessions (
    token         TEXT PRIMARY KEY,
    user_id       INTEGER NOT NULL,
    created_at    TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    expires_at    TEXT NOT NULL,
    revoked_at    TEXT,
    FOREIGN KEY(user_id) REFERENCES users(id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_sessions_user ON sessions(user_id);
CREATE INDEX IF NOT EXISTS idx_sessions_expiry ON sessions(expires_at);
"""


def init_db() -> None:
    """Create DB file + tables if missing. Safe to call repeatedly."""
    global _initialized
    with _init_lock:
        if _initialized:
            return
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(DB_PATH)
        try:
            conn.execute("PRAGMA journal_mode=WAL;")
            conn.execute("PRAGMA foreign_keys=ON;")
            conn.executescript(SCHEMA)
            conn.commit()
        finally:
            conn.close()
        _initialized = True
        logger.info(f"[db] Ready at {DB_PATH}")


@contextmanager
def conn() -> Iterator[sqlite3.Connection]:
    """Context-managed connection. Enables foreign keys + returns dict-like rows."""
    if not _initialized:
        init_db()
    c = sqlite3.connect(DB_PATH, timeout=10.0)
    try:
        c.execute("PRAGMA foreign_keys=ON;")
        c.row_factory = sqlite3.Row
        yield c
        c.commit()
    except Exception:
        c.rollback()
        raise
    finally:
        c.close()
