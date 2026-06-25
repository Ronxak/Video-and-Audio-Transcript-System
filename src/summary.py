"""
AI summarisation of a finished transcript via Groq.

Given the plain transcript text, this asks a Groq-hosted LLM for the key points
as a short bullet list and parses the reply back into a clean list of strings
for the UI. It's deliberately small and separate from ``translation`` so each
feature stays focused and independently testable.

Summarisation always uses Groq (see :func:`src.auth.get_groq_client`); it's a
new feature with no legacy provider to stay compatible with.
"""
from __future__ import annotations

import re
from typing import List, Optional

from . import auth, config


class SummaryError(RuntimeError):
    """Raised for configuration problems (e.g. a missing API key)."""


# Kept close to the wording requested for the feature; the system prompt nudges
# the model to return a clean, parse-friendly bullet list and nothing else.
_PROMPT = (
    "Please summarize the key points of the following video transcript in 5-7 "
    "bullet points:\n\n{transcript}"
)
_SYSTEM_PROMPT = (
    "You are a concise note-taker. Reply with ONLY the bullet points — one per "
    "line, each starting with '- '. No preamble, no title, no closing remarks."
)

# Leading bullet/numbering markers to strip when parsing the reply: "-", "*",
# "•", "1.", "1)" (optionally indented).
_BULLET_RE = re.compile(r"^\s*(?:[-*•‣◦]|\d+[.)])\s+")


def _parse_bullets(text: str) -> List[str]:
    """Turn a model reply into a clean list of bullet strings (no markers).

    If the reply contains explicit bullet/number markers we keep only those
    lines (dropping any stray preamble like "Here are the key points:"). If it
    has no markers at all, we fall back to treating each non-empty line as a
    point so we never silently drop a valid summary.
    """
    lines = [ln.strip() for ln in (text or "").splitlines() if ln.strip()]
    marked = [_BULLET_RE.sub("", ln).strip() for ln in lines if _BULLET_RE.match(ln)]
    marked = [m for m in marked if m]
    return marked if marked else lines


def summarize_transcript(
    transcript_text: str, model: Optional[str] = None
) -> List[str]:
    """Summarise a transcript into 5-7 key-point bullets.

    Args:
        transcript_text: the raw (plain) transcript text.
        model: optional Groq model override; defaults to ``config.SUMMARY_MODEL``.

    Returns:
        A list of bullet strings with leading markers removed.

    Raises:
        SummaryError: if the transcript is empty or no API key is configured.
    """
    text = (transcript_text or "").strip()
    if not text:
        raise SummaryError("Nothing to summarise — the transcript is empty.")

    model = model or config.SUMMARY_MODEL
    try:
        client = auth.get_groq_client()
    except auth.AuthError as exc:
        raise SummaryError(str(exc)) from exc

    # Groq exposes an OpenAI-compatible chat completions endpoint.
    response = client.chat.completions.create(
        model=model,
        max_tokens=1024,
        messages=[
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": _PROMPT.format(transcript=text)},
        ],
    )
    raw = response.choices[0].message.content or ""
    return _parse_bullets(raw)
