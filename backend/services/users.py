"""
backend/services/users.py

User CRUD + password hashing + settings helpers for the multi-tenant system.

Design decisions:
- Passwords stored as bcrypt hashes (salt included in hash string).
- User settings live in a separate table so we can extend without touching users.
- Openrouter key is stored as BLOB after AES-encryption (see keyvault.py in 5.4).
"""
from __future__ import annotations

import logging
from datetime import datetime
from typing import Optional, Dict, Any, List

import bcrypt

from .db import conn

logger = logging.getLogger(__name__)


class UserError(Exception):
    """Raised on any user-service business rule violation."""


def _hash_password(password: str) -> str:
    return bcrypt.hashpw(password.encode("utf-8"), bcrypt.gensalt(rounds=12)).decode("ascii")


def _verify_password(password: str, hashed: str) -> bool:
    try:
        return bcrypt.checkpw(password.encode("utf-8"), hashed.encode("ascii"))
    except (ValueError, TypeError):
        return False


# ---------- CRUD ----------

def create_user(
    username: str,
    password: str,
    email: Optional[str] = None,
    is_admin: bool = False,
) -> Dict[str, Any]:
    """Create a new user. Raises UserError if username exists."""
    if not username or not username.strip():
        raise UserError("username required")
    if not password or len(password) < 4:
        raise UserError("password must be at least 4 characters")

    username = username.strip()
    with conn() as c:
        existing = c.execute("SELECT id FROM users WHERE username = ?", (username,)).fetchone()
        if existing:
            raise UserError(f"username '{username}' already exists")
        cursor = c.execute(
            "INSERT INTO users (username, email, password_hash, is_admin) VALUES (?, ?, ?, ?)",
            (username, (email or "").strip() or None, _hash_password(password), int(bool(is_admin))),
        )
        user_id = cursor.lastrowid
        # Bootstrap default settings
        c.execute(
            "INSERT INTO user_settings (user_id, default_preset) VALUES (?, ?)",
            (user_id, "hybrid"),
        )
    logger.info(f"[users] created user_id={user_id} username={username} is_admin={is_admin}")
    return get_user_by_id(user_id)


def get_user_by_id(user_id: int) -> Optional[Dict[str, Any]]:
    with conn() as c:
        row = c.execute(
            "SELECT id, username, email, is_admin, created_at, disabled_at FROM users WHERE id = ?",
            (user_id,),
        ).fetchone()
        return dict(row) if row else None


def get_user_by_username(username: str) -> Optional[Dict[str, Any]]:
    with conn() as c:
        row = c.execute(
            "SELECT id, username, email, is_admin, created_at, disabled_at FROM users WHERE username = ? COLLATE NOCASE",
            (username.strip(),),
        ).fetchone()
        return dict(row) if row else None


def verify_credentials(username: str, password: str) -> Optional[Dict[str, Any]]:
    """Return the user record on success, None on any failure."""
    with conn() as c:
        row = c.execute(
            "SELECT id, username, email, password_hash, is_admin, disabled_at FROM users WHERE username = ? COLLATE NOCASE",
            (username.strip(),),
        ).fetchone()
        if not row:
            return None
        if row["disabled_at"]:
            return None
        if not _verify_password(password, row["password_hash"]):
            return None
        user = dict(row)
        user.pop("password_hash", None)
        return user


def list_users() -> List[Dict[str, Any]]:
    with conn() as c:
        rows = c.execute(
            "SELECT id, username, email, is_admin, created_at, disabled_at FROM users ORDER BY created_at ASC"
        ).fetchall()
        return [dict(r) for r in rows]


def set_password(user_id: int, new_password: str) -> None:
    if not new_password or len(new_password) < 4:
        raise UserError("password must be at least 4 characters")
    with conn() as c:
        c.execute("UPDATE users SET password_hash = ? WHERE id = ?", (_hash_password(new_password), user_id))


def set_disabled(user_id: int, disabled: bool) -> None:
    with conn() as c:
        c.execute(
            "UPDATE users SET disabled_at = ? WHERE id = ?",
            (datetime.utcnow().isoformat() if disabled else None, user_id),
        )


# ---------- Settings ----------

def get_settings(user_id: int) -> Dict[str, Any]:
    with conn() as c:
        row = c.execute(
            "SELECT user_id, default_preset, openrouter_key_encrypted, updated_at FROM user_settings WHERE user_id = ?",
            (user_id,),
        ).fetchone()
        if not row:
            return {"user_id": user_id, "default_preset": "hybrid", "openrouter_key_encrypted": None, "updated_at": None}
        return dict(row)


def set_default_preset(user_id: int, preset: str) -> None:
    if preset not in ("air_gapped", "hybrid", "cloud_premium"):
        raise UserError(f"invalid preset: {preset}")
    with conn() as c:
        c.execute(
            "UPDATE user_settings SET default_preset = ?, updated_at = CURRENT_TIMESTAMP WHERE user_id = ?",
            (preset, user_id),
        )


def set_openrouter_key(user_id: int, key_encrypted: Optional[bytes]) -> None:
    """Store an already-encrypted OpenRouter key (encryption handled by keyvault.py)."""
    with conn() as c:
        c.execute(
            "UPDATE user_settings SET openrouter_key_encrypted = ?, updated_at = CURRENT_TIMESTAMP WHERE user_id = ?",
            (key_encrypted, user_id),
        )
