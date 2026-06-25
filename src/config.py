"""
Central configuration.

Every value can be overridden with an environment variable (optionally placed in
a `.env` file at the project root). Defaults are chosen so the app runs well on a
GPU machine out of the box, while still falling back gracefully to CPU.
"""
from __future__ import annotations

import os
from dataclasses import dataclass

# Load a local .env file if present (handy for the API key). Optional dependency.
try:
    from dotenv import load_dotenv

    load_dotenv()
except Exception:  # pragma: no cover - dotenv is optional
    pass


def _detect_device() -> tuple[str, str]:
    """Return (device, compute_type).

    Prefers GPU when one is visible. We probe via CTranslate2 (which
    faster-whisper already depends on) so we don't pull in torch just to
    check for CUDA.
    """
    forced = os.getenv("DEVICE")
    if forced:
        default_ct = "float16" if forced == "cuda" else "int8"
        return forced, os.getenv("COMPUTE_TYPE", default_ct)
    try:
        import ctranslate2

        if ctranslate2.get_cuda_device_count() > 0:
            return "cuda", os.getenv("COMPUTE_TYPE", "float16")
    except Exception:
        pass
    return "cpu", os.getenv("COMPUTE_TYPE", "int8")


DEVICE, COMPUTE_TYPE = _detect_device()

# faster-whisper model. "large-v3" is the most accurate; on a weak CPU you may
# prefer "medium" or "small" (set WHISPER_MODEL accordingly).
WHISPER_MODEL = os.getenv("WHISPER_MODEL", "large-v3")

# ---- LLM enhancement: translation + summarisation ---------------------------
# Provider for the LLM-backed features. "groq" (default) uses Groq's fast,
# OpenAI-compatible API; "anthropic" uses the Claude API and is kept for
# backwards compatibility. API *keys* are resolved in src/auth.py (from the
# environment locally, or services.json on the Claude Code platform).
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "groq").strip().lower()

# Default model per provider (override with GROQ_MODEL / ANTHROPIC_MODEL).
# llama-3.3-70b-versatile is a strong general model on Groq; switch to
# llama-3.1-8b-instant for lower latency/cost.
GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
# Anthropic (legacy) — used only when LLM_PROVIDER=anthropic. Sonnet gives a
# good fluency/cost balance; claude-opus-4-8 is highest quality.
ANTHROPIC_MODEL = os.getenv("ANTHROPIC_MODEL", "claude-sonnet-4-6")

# Model used for translation, resolved for the active provider. The historical
# TRANSLATION_MODEL env var is still honoured as an explicit override (set it to
# a model that matches LLM_PROVIDER).
TRANSLATION_MODEL = os.getenv(
    "TRANSLATION_MODEL",
    ANTHROPIC_MODEL if LLM_PROVIDER == "anthropic" else GROQ_MODEL,
)
# Summarisation always runs on Groq, so it has its own Groq model setting.
SUMMARY_MODEL = os.getenv("SUMMARY_MODEL", GROQ_MODEL)

# ---- Safety limits ----------------------------------------------------------
MAX_FILE_SIZE_MB = int(os.getenv("MAX_FILE_SIZE_MB", "500"))


@dataclass(frozen=True)
class SubtitleStyle:
    """Professional subtitling constraints (broadcast / streaming standard).

    These are the rules real subtitlers follow; applying them is what turns a
    raw transcript into a *usable* subtitle file.
    """

    max_line_length: int = 42      # characters per line
    max_lines: int = 2             # lines per cue
    max_cps: float = 17.0          # characters per second (reading speed)
    min_duration: float = 1.0      # seconds a cue stays on screen (minimum)
    max_duration: float = 7.0      # seconds a cue stays on screen (maximum)
    min_gap: float = 0.084         # ~2 frames @ 24fps gap between consecutive cues


SUBTITLE_STYLE = SubtitleStyle()

# Languages offered in the "spoken language" dropdown (Whisper supports ~99).
# Indian languages are listed first because they are the primary use case here.
SUPPORTED_LANGUAGES: dict[str, str | None] = {
    "Auto-detect": None,
    "Hindi": "hi",
    "Bengali": "bn",
    "Tamil": "ta",
    "Telugu": "te",
    "Marathi": "mr",
    "Gujarati": "gu",
    "Kannada": "kn",
    "Malayalam": "ml",
    "Punjabi": "pa",
    "Urdu": "ur",
    "English": "en",
    "Spanish": "es",
    "French": "fr",
    "German": "de",
    "Arabic": "ar",
    "Chinese": "zh",
    "Japanese": "ja",
}

# Target languages offered for the translation enhancement.
TARGET_LANGUAGES: list[str] = [
    "English", "Hindi", "Bengali", "Tamil", "Telugu", "Marathi", "Gujarati",
    "Kannada", "Malayalam", "Punjabi", "Urdu", "Spanish", "French", "German",
    "Arabic", "Chinese", "Japanese",
]
