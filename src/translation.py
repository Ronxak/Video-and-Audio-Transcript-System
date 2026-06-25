"""
Subtitle translation via an LLM.

By default this uses **Groq** (fast, OpenAI-compatible). The Anthropic Claude
API is still supported for backwards compatibility — set ``LLM_PROVIDER=anthropic``
to use it. The client is obtained from :mod:`src.auth` either way, so this module
never touches API keys directly.

Why an LLM rather than a classic NMT model: subtitle translation benefits from
context and tone awareness, and an instruction-tuned model produces the most
natural, on-screen-readable output across many language pairs with zero
per-language setup.

Design notes:
- Lines are translated in batches (one API call per batch) to stay efficient
  while keeping a strict 1:1 alignment with the original cues (so timestamps
  never drift).
- The model is asked to return a JSON array of strings; parsing is defensive
  and falls back to the original text if anything is off, so the pipeline never
  hard-crashes mid-job.
"""
from __future__ import annotations

import json
import re
from typing import Callable, List, Optional

from . import auth, config
from .srt_utils import Cue


class TranslationError(RuntimeError):
    """Raised for configuration problems (e.g. a missing API key)."""


_SYSTEM_PROMPT = (
    "You are a professional subtitle translator at a localization studio. "
    "You will receive a JSON array of {n} subtitle lines in the source language. "
    "Translate EVERY line into {target} — never leave a line in the original "
    "language. Keep each translation concise and natural for on-screen reading, "
    "faithful to the original meaning and tone, with no notes or commentary. "
    'Respond with a single JSON object of the form {{"translations": [...]}} whose '
    "value is an array of exactly {n} translated strings, in the same order as the "
    "input. Output valid json only."
)


def _get_client():
    """Return the LLM client for the active provider (Groq by default)."""
    try:
        if config.LLM_PROVIDER == "anthropic":
            return auth.get_anthropic_client()
        return auth.get_groq_client()
    except auth.AuthError as exc:
        # Preserve this module's historical error type for callers/tests.
        raise TranslationError(str(exc)) from exc


def _parse_translations(raw: str, n: int) -> Optional[List[str]]:
    """Best-effort: extract a list of exactly ``n`` translated strings, else None.

    Tolerates the several shapes models actually return: a bare JSON array, a
    ``{"translations": [...]}`` object, an object with a single list value under
    any key (e.g. a singular ``"translation"``), an index-keyed object
    (``{"0": "...", "1": "..."}``), and chatty text with JSON embedded.
    """
    def _as_list(obj):
        if isinstance(obj, list):
            return obj
        if isinstance(obj, dict):
            if isinstance(obj.get("translations"), list):
                return obj["translations"]
            list_values = [v for v in obj.values() if isinstance(v, list)]
            if len(list_values) == 1:
                return list_values[0]
            try:  # index-keyed object: {"0": "...", "1": "..."}
                items = sorted(obj.items(), key=lambda kv: int(kv[0]))
                if items and all(isinstance(v, str) for _, v in items):
                    return [v for _, v in items]
            except (ValueError, TypeError):
                pass
        return None

    candidates = []
    try:
        candidates.append(json.loads(raw))
    except (json.JSONDecodeError, TypeError):
        match = re.search(r"\{.*\}|\[.*\]", raw or "", re.DOTALL)
        if match:
            try:
                candidates.append(json.loads(match.group(0)))
            except json.JSONDecodeError:
                pass
    for obj in candidates:
        lst = _as_list(obj)
        if isinstance(lst, list) and len(lst) == n:
            return [str(x) for x in lst]
    return None


def _complete(system: str, user: str, model: str) -> str:
    """Single-turn completion that returns the model's text, per provider."""
    client = _get_client()
    if config.LLM_PROVIDER == "anthropic":
        response = client.messages.create(
            model=model,
            max_tokens=4096,
            system=system,
            messages=[{"role": "user", "content": user}],
        )
        return "".join(
            block.text for block in response.content
            if getattr(block, "type", None) == "text"
        )
    # Groq (default): OpenAI-compatible chat completions.
    response = client.chat.completions.create(
        model=model,
        max_tokens=4096,
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
    )
    return response.choices[0].message.content or ""


def _translate_batch(
    texts: List[str], target_language: str, model: str, _depth: int = 0
) -> List[str]:
    """Translate one batch; on a shape/count mismatch, split and retry.

    Smaller chunks are far more reliable (the model rarely miscounts a handful of
    lines), so rather than silently dropping a whole batch back to the originals
    we recurse down toward single lines, only keeping an original if even a
    one-line request can't be parsed.
    """
    if not texts:
        return []

    raw = _complete(
        _SYSTEM_PROMPT.format(target=target_language, n=len(texts)),
        json.dumps(texts, ensure_ascii=False),
        model,
    )
    parsed = _parse_translations(raw, len(texts))
    if parsed is not None:
        return parsed

    if len(texts) > 1 and _depth < 6:
        mid = len(texts) // 2
        return (
            _translate_batch(texts[:mid], target_language, model, _depth + 1)
            + _translate_batch(texts[mid:], target_language, model, _depth + 1)
        )
    # Even a single line failed to parse → keep the original to preserve timing.
    return texts


def translate_cues(
    cues: List[Cue],
    target_language: str,
    model: Optional[str] = None,
    batch_size: int = 40,
    progress: Optional[Callable[[float], None]] = None,
) -> List[Cue]:
    """Translate every cue's text into ``target_language`` (in place)."""
    model = model or config.TRANSLATION_MODEL
    # Flatten any in-cue line breaks so the model sees clean sentences.
    texts = [" ".join(cue.text.split()) for cue in cues]

    translated: List[str] = []
    for start in range(0, len(texts), batch_size):
        chunk = texts[start:start + batch_size]
        translated.extend(_translate_batch(chunk, target_language, model))
        if progress:
            progress(min(1.0, (start + len(chunk)) / max(1, len(texts))))

    for cue, text in zip(cues, translated):
        cue.translation = text
    return cues
