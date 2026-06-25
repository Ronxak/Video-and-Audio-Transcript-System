"""
End-to-end pipeline: media file -> transcript -> SRT (+ optional translation).

Keeping the orchestration in one place keeps `app.py` (the UI) thin and makes
the core flow reusable from a CLI, a test, or another service.
"""
from __future__ import annotations

import os
import tempfile
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional

from . import config, srt_utils, summary, transcription, translation
from .srt_utils import Cue

VIDEO_EXTENSIONS = {
    ".mp4", ".mkv", ".mov", ".avi", ".webm", ".m4v", ".flv", ".mpeg", ".mpg",
}


@dataclass
class Result:
    cues: List[Cue]
    detected_language: str
    duration: float
    srt_path: str
    vtt_path: str
    txt_path: str
    is_video: bool
    stats: Dict = field(default_factory=dict)
    translated_srt_path: Optional[str] = None
    bilingual_srt_path: Optional[str] = None
    target_language: Optional[str] = None
    summary: Optional[List[str]] = None


def process(
    media_path: str,
    language: Optional[str] = None,
    translate_to: Optional[str] = None,
    summarize: bool = False,
    progress: Optional[Callable[[float, str], None]] = None,
) -> Result:
    """Run the full pipeline and write output files to a temp directory."""

    def report(fraction: float, message: str = "Working…") -> None:
        if progress:
            progress(fraction, message)

    # ---- Validate input ----------------------------------------------------
    if not media_path or not os.path.exists(media_path):
        raise FileNotFoundError("No media file provided.")
    size_mb = os.path.getsize(media_path) / (1024 * 1024)
    if size_mb > config.MAX_FILE_SIZE_MB:
        raise ValueError(
            f"File is {size_mb:.0f} MB, above the {config.MAX_FILE_SIZE_MB} MB limit."
        )

    # ---- Transcribe --------------------------------------------------------
    report(0.03, "Loading model…")
    cues, detected, duration = transcription.transcribe(
        media_path,
        language=language,
        progress=(lambda f, m: report(0.05 + 0.70 * f, m)) if progress else None,
    )
    if not cues:
        raise ValueError("No speech could be detected in this file.")

    # ---- Apply professional subtitle formatting ----------------------------
    # 1) Split over-long segments into readable cues, then 2) fix timing.
    cues = srt_utils.split_long_cues(cues, config.SUBTITLE_STYLE)
    srt_utils.enforce_timing(cues, config.SUBTITLE_STYLE)

    style = config.SUBTITLE_STYLE
    out_dir = tempfile.mkdtemp(prefix="transcript_")
    base = os.path.splitext(os.path.basename(media_path))[0] or "transcript"

    srt_path = os.path.join(out_dir, f"{base}.srt")
    vtt_path = os.path.join(out_dir, f"{base}.vtt")
    txt_path = os.path.join(out_dir, f"{base}.txt")
    srt_utils.write_text_file(srt_path, srt_utils.cues_to_srt(cues, style))
    srt_utils.write_text_file(vtt_path, srt_utils.cues_to_vtt(cues, style))
    srt_utils.write_text_file(txt_path, srt_utils.cues_to_text(cues))

    result = Result(
        cues=cues,
        detected_language=detected,
        duration=duration,
        srt_path=srt_path,
        vtt_path=vtt_path,
        txt_path=txt_path,
        is_video=os.path.splitext(media_path)[1].lower() in VIDEO_EXTENSIONS,
    )

    # ---- Optional translation (the enhancement) ----------------------------
    if translate_to:
        report(0.80, f"Translating to {translate_to}…")
        translation.translate_cues(
            cues,
            translate_to,
            progress=(lambda f: report(0.80 + 0.18 * f, f"Translating to {translate_to}…"))
            if progress
            else None,
        )
        translated_path = os.path.join(out_dir, f"{base}.{translate_to.lower()}.srt")
        bilingual_path = os.path.join(out_dir, f"{base}.bilingual.srt")
        srt_utils.write_text_file(
            translated_path, srt_utils.cues_to_srt(cues, style, use_translation=True)
        )
        srt_utils.write_text_file(
            bilingual_path, srt_utils.cues_to_srt(cues, style, bilingual=True)
        )
        result.translated_srt_path = translated_path
        result.bilingual_srt_path = bilingual_path
        result.target_language = translate_to

    # ---- Optional AI summarisation (Groq) ----------------------------------
    if summarize:
        report(0.98, "Summarizing…")
        result.summary = summary.summarize_transcript(srt_utils.cues_to_text(cues))

    result.stats = srt_utils.compute_stats(cues)
    report(1.0, "Done")
    return result
