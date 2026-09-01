"""
backend/services/keyvault.py

Small AES-256-GCM helper for encrypting per-user secrets (like OpenRouter
API keys) before storing them in SQLite.

Master key resolution (first hit wins):
1. env VAULT_MASTER_KEY  — 32 bytes, base64-url encoded
2. file backend/data/vault.key  — auto-generated on first use (0600 perms
   on POSIX; readable-by-owner-only). Persists across container restarts
   because backend/data is volume-mounted.

If neither source has a key, one is generated and written to (2) so the
system self-bootstraps on first run. Losing this file makes previously
stored ciphertexts unrecoverable — treat it like a database key.

Ciphertext layout: 12-byte nonce || GCM ciphertext+tag.
"""
from __future__ import annotations

import os
import base64
import secrets
import logging
import threading
from pathlib import Path
from typing import Optional

from cryptography.hazmat.primitives.ciphers.aead import AESGCM

from .db import DATA_DIR  # reuse the same data dir as users.db

logger = logging.getLogger(__name__)

_MASTER_LOCK = threading.Lock()
_MASTER: Optional[bytes] = None
KEY_PATH = DATA_DIR / "vault.key"
_NONCE_BYTES = 12


def _load_or_create_master() -> bytes:
    global _MASTER
    with _MASTER_LOCK:
        if _MASTER is not None:
            return _MASTER

        # 1. Env var
        env_val = (os.getenv("VAULT_MASTER_KEY") or "").strip()
        if env_val:
            try:
                key = base64.urlsafe_b64decode(env_val + "=" * (-len(env_val) % 4))
                if len(key) != 32:
                    raise ValueError(f"VAULT_MASTER_KEY must decode to 32 bytes (got {len(key)})")
                _MASTER = key
                logger.info("[keyvault] master key loaded from env VAULT_MASTER_KEY")
                return key
            except Exception as e:
                logger.error(f"[keyvault] VAULT_MASTER_KEY invalid: {e} — falling back to file")

        # 2. File on disk (auto-generate first time)
        KEY_PATH.parent.mkdir(parents=True, exist_ok=True)
        if KEY_PATH.exists():
            raw = KEY_PATH.read_bytes().strip()
            try:
                key = base64.urlsafe_b64decode(raw + b"=" * (-len(raw) % 4))
                if len(key) != 32:
                    raise ValueError(f"vault.key must be 32 bytes decoded (got {len(key)})")
                _MASTER = key
                logger.info(f"[keyvault] master key loaded from {KEY_PATH}")
                return key
            except Exception as e:
                logger.error(f"[keyvault] vault.key unreadable: {e} — regenerating (OLD SECRETS ARE LOST)")

        # Generate + persist
        new_key = secrets.token_bytes(32)
        b64 = base64.urlsafe_b64encode(new_key).rstrip(b"=")
        KEY_PATH.write_bytes(b64)
        try:
            os.chmod(KEY_PATH, 0o600)  # POSIX only
        except Exception:
            pass  # Windows / non-POSIX — best-effort
        _MASTER = new_key
        logger.warning(f"[keyvault] generated new master key at {KEY_PATH} — back this file up.")
        return new_key


def encrypt(plaintext: str) -> bytes:
    """Encrypt a string with AES-256-GCM. Returns nonce||ciphertext bytes."""
    if plaintext is None:
        raise ValueError("plaintext required")
    key = _load_or_create_master()
    aes = AESGCM(key)
    nonce = secrets.token_bytes(_NONCE_BYTES)
    ct = aes.encrypt(nonce, plaintext.encode("utf-8"), associated_data=None)
    return nonce + ct


def decrypt(blob: bytes) -> Optional[str]:
    """Decrypt a blob written by encrypt(). Returns None on any error."""
    if not blob or len(blob) <= _NONCE_BYTES:
        return None
    try:
        key = _load_or_create_master()
        aes = AESGCM(key)
        nonce, ct = blob[:_NONCE_BYTES], blob[_NONCE_BYTES:]
        return aes.decrypt(nonce, ct, associated_data=None).decode("utf-8")
    except Exception as e:
        logger.warning(f"[keyvault] decrypt failed: {e}")
        return None
