"""
Speech-to-text using faster-whisper.

faster-whisper is a CTranslate2 re-implementation of OpenAI Whisper: same
accuracy, ~4x faster, lower memory, with word/segment timestamps and built-in
audio decoding (so it reads the audio track straight out of a video file).
"""
from __future__ import annotations

import os
from functools import lru_cache
from typing import Callable, List, Optional, Tuple

from . import config
from .srt_utils import Cue


@lru_cache(maxsize=1)
def get_model():
    """Load (and cache) the Whisper model. Heavy import is done lazily."""
    from faster_whisper import WhisperModel

    return WhisperModel(
        config.WHISPER_MODEL,
        device=config.DEVICE,
        compute_type=config.COMPUTE_TYPE,
    )


def transcribe(
    media_path: str,
    language: Optional[str] = None,
    progress: Optional[Callable[[float, str], None]] = None,
) -> Tuple[List[Cue], str, float]:
    """Transcribe an audio or video file.

    Args:
        media_path: path to an audio or video file.
        language: ISO code (e.g. ``"hi"``) or ``None`` to auto-detect.
        progress: optional callback ``(fraction_0_to_1, message)``.

    Returns:
        ``(cues, detected_language, duration_seconds)``.
    """
    if not media_path or not os.path.exists(media_path):
        raise FileNotFoundError("Media file not found.")

    model = get_model()

    # `transcribe` returns a generator; iterating it is what does the work.
    segments_gen, info = model.transcribe(
        media_path,
        language=language,        # None => auto-detect
        task="transcribe",
        vad_filter=True,          # drop long silences -> cleaner cue timing
        beam_size=5,
    )

    duration = float(getattr(info, "duration", 0.0) or 0.0)
    detected = getattr(info, "language", None) or (language or "unknown")

    cues: List[Cue] = []
    for segment in segments_gen:
        text = (segment.text or "").strip()
        if not text:
            continue
        cues.append(Cue(start=float(segment.start), end=float(segment.end), text=text))
        if progress and duration > 0:
            progress(min(0.99, segment.end / duration), "Transcribing…")

    return cues, detected, duration
