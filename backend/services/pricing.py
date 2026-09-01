"""
backend/services/pricing.py

Static price table for per-token billing (USD per 1M tokens).
Prices reference public rate cards as of 2026-Q3; update in one place when
providers change them.

Local Ollama models cost $0 — they run on the on-server GPU.

Rate format: (input_usd_per_million, output_usd_per_million)
"""
from __future__ import annotations

from typing import Optional, Tuple

# USD per 1,000,000 tokens
PRICING: dict[str, Tuple[float, float]] = {
    # ---- OpenRouter / Google (Gemini) ----
    "google/gemini-2.5-flash":        (0.075,  0.30),
    "google/gemini-2.5-flash-lite":   (0.037,  0.15),
    "google/gemini-2.5-pro":          (1.25,   5.00),
    "google/gemini-2.0-flash":        (0.10,   0.40),

    # ---- OpenRouter / Anthropic (Claude) ----
    "anthropic/claude-sonnet-4.5":    (3.0,    15.0),
    "anthropic/claude-opus-4.7":      (15.0,   75.0),
    "anthropic/claude-3.5-haiku":     (0.80,   4.0),

    # ---- OpenRouter / OpenAI ----
    "openai/gpt-4o":                  (2.50,   10.0),
    "openai/gpt-4o-mini":             (0.15,   0.60),

    # ---- Local Ollama (free — runs on-server GPU) ----
    "qwen2.5:14b":                    (0.0,    0.0),
    "qwen2.5:14b-gpu":                (0.0,    0.0),
    "qwen2.5:7b":                     (0.0,    0.0),
    "qwen2.5:7b-fast":                (0.0,    0.0),
    "qwen2.5:7b-instruct-q4_K_M":     (0.0,    0.0),
    "qwen2.5:1.5b":                   (0.0,    0.0),
    "bge-m3":                         (0.0,    0.0),
    "bge-m3:latest":                  (0.0,    0.0),
}

_FALLBACK_RATE = (0.0, 0.0)  # unknown model → treat as free (better than misreport)


def _lookup_rate(model: str) -> Tuple[float, float]:
    """Case-tolerant lookup — providers sometimes vary casing/prefix.
    Also strips common openai/ prefix that litellm adds.
    """
    if not model:
        return _FALLBACK_RATE
    m = model.strip()
    if m in PRICING:
        return PRICING[m]
    # Strip litellm 'openai/' prefix
    if m.startswith("openai/") and m[7:] in PRICING:
        return PRICING[m[7:]]
    # Case-insensitive fallback
    lower_map = {k.lower(): v for k, v in PRICING.items()}
    return lower_map.get(m.lower(), _FALLBACK_RATE)


def calculate_cost(model: str, prompt_tokens: int, completion_tokens: int) -> float:
    """Return estimated USD cost for a single call."""
    rate_in, rate_out = _lookup_rate(model)
    return (
        (prompt_tokens or 0) * rate_in / 1_000_000.0 +
        (completion_tokens or 0) * rate_out / 1_000_000.0
    )


def is_known_model(model: str) -> bool:
    return _lookup_rate(model) != _FALLBACK_RATE or (model and model.lower() in {k.lower() for k in PRICING})
