"""
backend/services/sessions.py

Opaque session tokens stored in the sessions table.

Design:
- Login issues a 48-character URL-safe random token.
- Token → user_id lookup via DB, with expiry + revoked_at columns.
- Cheap validation: SELECT + expiry check + revoked check.
- Logout marks revoked_at.
- verify_token() is called on every authed request by middleware.
"""
from __future__ import annotations

import os
import secrets
import logging
from datetime import datetime, timedelta
from typing import Optional, Dict, Any

from .db import conn

logger = logging.getLogger(__name__)

SESSION_TTL_HOURS = int(os.getenv("SESSION_TTL_HOURS", "168"))  # default: 7 days
_TOKEN_BYTES = 36  # → 48 URL-safe chars


def create_session(user_id: int, ttl_hours: Optional[int] = None) -> Dict[str, Any]:
    """Issue a new opaque session token for the user."""
    token = secrets.token_urlsafe(_TOKEN_BYTES)
    ttl = ttl_hours if ttl_hours is not None else SESSION_TTL_HOURS
    expires_at = datetime.utcnow() + timedelta(hours=ttl)
    with conn() as c:
        c.execute(
            "INSERT INTO sessions (token, user_id, expires_at) VALUES (?, ?, ?)",
            (token, user_id, expires_at.isoformat()),
        )
    return {
        "token": token,
        "user_id": user_id,
        "expires_at": expires_at.isoformat() + "Z",
    }


def verify_token(token: str) -> Optional[Dict[str, Any]]:
    """Look up a session token; returns {user_id, expires_at} on success.
    None if token is unknown, expired, or revoked.
    """
    if not token:
        return None
    with conn() as c:
        row = c.execute(
            "SELECT token, user_id, expires_at, revoked_at FROM sessions WHERE token = ?",
            (token,),
        ).fetchone()
        if not row:
            return None
        if row["revoked_at"]:
            return None
        try:
            exp = datetime.fromisoformat(row["expires_at"])
        except (ValueError, TypeError):
            return None
        if exp < datetime.utcnow():
            return None
        return {"user_id": int(row["user_id"]), "expires_at": row["expires_at"]}


def revoke_token(token: str) -> bool:
    """Mark a session revoked. Returns True if a row was updated."""
    if not token:
        return False
    with conn() as c:
        cur = c.execute(
            "UPDATE sessions SET revoked_at = CURRENT_TIMESTAMP WHERE token = ? AND revoked_at IS NULL",
            (token,),
        )
        return cur.rowcount > 0


def revoke_all_for_user(user_id: int) -> int:
    """Revoke every non-revoked session for a user (e.g. on password change)."""
    with conn() as c:
        cur = c.execute(
            "UPDATE sessions SET revoked_at = CURRENT_TIMESTAMP WHERE user_id = ? AND revoked_at IS NULL",
            (user_id,),
        )
        return cur.rowcount


def cleanup_expired(older_than_hours: int = 24 * 30) -> int:
    """Delete very old expired sessions to keep the table tidy."""
    cutoff = (datetime.utcnow() - timedelta(hours=older_than_hours)).isoformat()
    with conn() as c:
        cur = c.execute("DELETE FROM sessions WHERE expires_at < ?", (cutoff,))
        return cur.rowcount
