"""
backend/services/model_router.py

Central resolver for answer-LLM configuration.

Two modes are supported per request:
- "local"  → the on-server Ollama instance (default main model, GPU-offloaded)
- "api"    → a cloud provider via VISION_API_BASE / VISION_API_KEY
             (defaults to google/gemini-2.5-flash — same account used for OCR)

Callers pick a mode per /ask call. Query rewriter, planner, and grader keep
their own configs (they use fast small models and don't need to switch).

Env overrides:
- CLOUD_LLM_MODEL      (default: google/gemini-2.5-flash)
- CLOUD_LLM_API_BASE   (default: VISION_API_BASE)
- CLOUD_LLM_API_KEY    (default: VISION_API_KEY)
- DEFAULT_LLM_MODE     (auto|local|api; default: auto → local)
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional

from ingestion.config import CUSTOM_API_BASE, CUSTOM_API_KEY, CUSTOM_MODEL_NAME

LLM_MODE_AUTO = "auto"
LLM_MODE_LOCAL = "local"
LLM_MODE_API = "api"
LLM_MODES = (LLM_MODE_AUTO, LLM_MODE_LOCAL, LLM_MODE_API)


@dataclass
class LLMConfig:
    """Resolved config the answer LLM call should use."""
    provider: str        # "local" or "api" (for logging/telemetry)
    model: str           # model id (litellm openai/{name})
    api_base: str
    api_key: str


def _cloud_defaults():
    return {
        "model": os.getenv("CLOUD_LLM_MODEL", "google/gemini-2.5-flash"),
        "api_base": (os.getenv("CLOUD_LLM_API_BASE") or os.getenv("VISION_API_BASE") or "https://openrouter.ai/api/v1").rstrip("/"),
        "api_key": os.getenv("CLOUD_LLM_API_KEY") or os.getenv("VISION_API_KEY") or "",
    }


def resolve_llm(mode: Optional[str] = None) -> LLMConfig:
    """Return the LLM config for the requested mode.

    Fail-open: if "api" is requested but no cloud key is configured, falls
    back to "local" (never blocks a query on missing cloud creds).
    """
    mode = (mode or os.getenv("DEFAULT_LLM_MODE") or LLM_MODE_AUTO).lower().strip()
    if mode not in LLM_MODES:
        mode = LLM_MODE_AUTO

    if mode == LLM_MODE_AUTO:
        mode = LLM_MODE_LOCAL  # auto → local by default (matches historical behavior)

    if mode == LLM_MODE_API:
        cloud = _cloud_defaults()
        if not cloud["api_key"]:
            # Silent downgrade to local — caller sees provider="local" in response
            mode = LLM_MODE_LOCAL
        else:
            return LLMConfig(
                provider="api",
                model=cloud["model"],
                api_base=cloud["api_base"],
                api_key=cloud["api_key"],
            )

    # local
    return LLMConfig(
        provider="local",
        model=CUSTOM_MODEL_NAME,
        api_base=CUSTOM_API_BASE,
        api_key=CUSTOM_API_KEY or "ollama",
    )


def format_model_id(cfg: LLMConfig) -> str:
    """Format a model id the way litellm expects (openai/{name})."""
    if cfg.provider == "api":
        # OpenRouter / OpenAI-compatible: pass model name as-is (e.g. "google/gemini-2.5-flash")
        # litellm still needs a provider prefix so we route through the openai schema:
        return f"openai/{cfg.model}"
    return f"openai/{cfg.model}"
