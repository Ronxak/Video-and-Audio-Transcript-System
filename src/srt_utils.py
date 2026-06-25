"""
SRT / VTT / text generation and professional subtitle formatting.

This module is intentionally PURE (standard library only, no model or network
dependencies) so its logic can be unit-tested in isolation. The trickiest and
most failure-prone part of a transcription system is the subtitle formatting
itself (timestamp format, UTF-8 text, line length and reading-speed limits), so
it lives here and is covered by tests in `tests/test_srt_utils.py`.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, TYPE_CHECKING

if TYPE_CHECKING:  # keep this module import-light for pure testing
    from .config import SubtitleStyle


@dataclass
class Cue:
    """A single subtitle entry. `text` may contain '\\n' once line-wrapped."""

    start: float
    end: float
    text: str
    translation: Optional[str] = None


# --------------------------------------------------------------------------- #
# Timestamps
# --------------------------------------------------------------------------- #
def format_timestamp(seconds: float, *, vtt: bool = False) -> str:
    """Format seconds as an SRT (``HH:MM:SS,mmm``) or VTT (``HH:MM:SS.mmm``) stamp.

    The comma-before-milliseconds is the single most common SRT mistake, so it
    is handled explicitly here.
    """
    if seconds is None or seconds < 0:
        seconds = 0.0
    total_ms = int(round(float(seconds) * 1000.0))
    hours, rem = divmod(total_ms, 3_600_000)
    minutes, rem = divmod(rem, 60_000)
    secs, ms = divmod(rem, 1000)
    sep = "." if vtt else ","
    return f"{hours:02d}:{minutes:02d}:{secs:02d}{sep}{ms:03d}"


# --------------------------------------------------------------------------- #
# Line wrapping
# --------------------------------------------------------------------------- #
def _balanced_two_line(text: str, max_line_length: int) -> Optional[str]:
    """Split text across two lines of similar length, if both fit. Else None."""
    words = text.split(" ")
    best: Optional[tuple[int, str, str]] = None
    for i in range(1, len(words)):
        line1 = " ".join(words[:i])
        line2 = " ".join(words[i:])
        if len(line1) <= max_line_length and len(line2) <= max_line_length:
            score = abs(len(line1) - len(line2))  # smaller = more balanced
            if best is None or score < best[0]:
                best = (score, line1, line2)
    return f"{best[1]}\n{best[2]}" if best else None


def _greedy_lines(text: str, max_line_length: int) -> List[str]:
    """Pack words into lines no longer than ``max_line_length`` (greedy)."""
    words = text.split()
    lines: List[str] = []
    current = ""
    for word in words:
        candidate = word if not current else f"{current} {word}"
        if len(candidate) <= max_line_length:
            current = candidate
        else:
            if current:
                lines.append(current)
            current = word
    if current:
        lines.append(current)
    return lines or ([text] if text else [])


def wrap_text(text: str, max_line_length: int = 42, max_lines: int = 2) -> str:
    """Wrap text to at most ``max_lines`` lines of ``max_line_length`` chars.

    For the common two-line case we prefer a *balanced* split (both lines a
    similar length), which reads better on screen. Falls back to greedy wrapping
    and, only if the text genuinely cannot fit, merges the overflow into the last
    line rather than ever dropping words (cue-splitting upstream prevents this in
    practice).
    """
    text = " ".join(text.split())  # normalise whitespace
    if not text:
        return ""
    if len(text) <= max_line_length:
        return text

    if max_lines == 2:
        balanced = _balanced_two_line(text, max_line_length)
        if balanced:
            return balanced

    lines = _greedy_lines(text, max_line_length)
    if len(lines) > max_lines:
        head = lines[: max_lines - 1]
        tail = " ".join(lines[max_lines - 1:])
        lines = head + [tail]
    return "\n".join(lines)


def _wrap_block(text: str, style: "SubtitleStyle") -> str:
    """Wrap each already-separated line of a block independently.

    Lines that are already within budget (e.g. produced by `split_long_cues`)
    pass through unchanged; a raw single line gets balance-wrapped.
    """
    return "\n".join(
        wrap_text(part, style.max_line_length, style.max_lines)
        for part in text.split("\n")
    )


# --------------------------------------------------------------------------- #
# Reading speed + timing compliance
# --------------------------------------------------------------------------- #
def compute_cps(text: str, start: float, end: float) -> float:
    """Characters-per-second reading speed for a cue."""
    chars = len(text.replace("\n", " "))
    duration = max(end - start, 1e-3)
    return chars / duration


def enforce_timing(cues: List[Cue], style: "SubtitleStyle") -> List[Cue]:
    """Adjust cue timing in place to satisfy subtitling standards.

    - Clamp negative starts to 0.
    - Cap each cue at ``max_duration``.
    - Extend too-short cues up to ``min_duration`` (without colliding with next).
    - Guarantee a minimum gap between consecutive cues.
    """
    cues.sort(key=lambda c: c.start)
    n = len(cues)

    for i, cue in enumerate(cues):
        start = max(0.0, cue.start)
        end = max(start, cue.end)

        if end - start > style.max_duration:
            end = start + style.max_duration

        if end - start < style.min_duration:
            desired = start + style.min_duration
            if i + 1 < n:
                room = cues[i + 1].start - style.min_gap
                end = max(start, min(desired, room))
            else:
                end = desired

        cue.start, cue.end = start, end

    for i in range(n - 1):
        if cues[i].end > cues[i + 1].start - style.min_gap:
            cues[i].end = max(cues[i].start, cues[i + 1].start - style.min_gap)

    return cues


def split_long_cues(cues: List[Cue], style: "SubtitleStyle") -> List[Cue]:
    """Split any cue that needs more than ``max_lines`` into several cues.

    Words are packed into lines within the per-line budget, then lines are
    grouped ``max_lines`` at a time into cues, and the cue's duration is shared
    proportionally to each group's length. This mirrors how professional
    subtitling tools keep long speech readable. Returns a NEW list.
    """
    result: List[Cue] = []
    for cue in cues:
        text = " ".join(cue.text.split())
        lines = _greedy_lines(text, style.max_line_length)

        if len(lines) <= style.max_lines:
            result.append(cue)  # serializer will balance-wrap if needed
            continue

        groups = [
            lines[i:i + style.max_lines]
            for i in range(0, len(lines), style.max_lines)
        ]
        total_chars = sum(len(" ".join(g)) for g in groups) or 1
        span = max(cue.end - cue.start, 1e-3)
        cursor = cue.start
        for i, group in enumerate(groups):
            is_last = i == len(groups) - 1
            end = cue.end if is_last else cursor + span * (len(" ".join(group)) / total_chars)
            result.append(Cue(start=cursor, end=end, text="\n".join(group)))
            cursor = end
    return result


# --------------------------------------------------------------------------- #
# Serialisation
# --------------------------------------------------------------------------- #
def _cue_text(cue: Cue, *, use_translation: bool, bilingual: bool) -> str:
    if bilingual and cue.translation:
        return f"{cue.text}\n{cue.translation}"
    if use_translation and cue.translation:
        return cue.translation
    return cue.text


def cues_to_srt(
    cues: List[Cue],
    style: "SubtitleStyle",
    *,
    use_translation: bool = False,
    bilingual: bool = False,
) -> str:
    """Render cues to an SRT string (sequential index, comma timestamps)."""
    blocks: List[str] = []
    for index, cue in enumerate(cues, start=1):
        raw = _cue_text(cue, use_translation=use_translation, bilingual=bilingual)
        stamp = f"{format_timestamp(cue.start)} --> {format_timestamp(cue.end)}"
        blocks.append(f"{index}\n{stamp}\n{_wrap_block(raw, style)}")
    return "\n\n".join(blocks) + "\n"


def cues_to_vtt(cues: List[Cue], style: "SubtitleStyle") -> str:
    """Render cues to a WebVTT string (period timestamps, WEBVTT header)."""
    blocks: List[str] = []
    for index, cue in enumerate(cues, start=1):
        stamp = (
            f"{format_timestamp(cue.start, vtt=True)} --> "
            f"{format_timestamp(cue.end, vtt=True)}"
        )
        blocks.append(f"{index}\n{stamp}\n{_wrap_block(cue.text, style)}")
    return "WEBVTT\n\n" + "\n\n".join(blocks) + "\n"


def cues_to_text(cues: List[Cue]) -> str:
    """Plain-text transcript (no timestamps, no mid-cue line breaks)."""
    return "\n".join(" ".join(c.text.split()) for c in cues) + "\n"


def preview_text(cues: List[Cue], limit: int = 60) -> str:
    """Human-readable preview with ``[mm:ss]`` markers for the UI."""
    lines: List[str] = []
    for cue in cues[:limit]:
        mm, ss = divmod(int(cue.start), 60)
        stamp = f"[{mm:02d}:{ss:02d}]"
        body = " ".join(cue.text.split())
        if cue.translation:
            lines.append(f"{stamp} {body}\n        > {cue.translation}")
        else:
            lines.append(f"{stamp} {body}")
    if len(cues) > limit:
        lines.append(f"... (+{len(cues) - limit} more cues in the downloaded file)")
    return "\n".join(lines)


def compute_stats(cues: List[Cue]) -> dict:
    """Summary statistics shown in the UI."""
    if not cues:
        return {"count": 0, "avg_cps": 0.0, "max_cps": 0.0, "total_duration": 0.0}
    cps_values = [compute_cps(c.text, c.start, c.end) for c in cues]
    return {
        "count": len(cues),
        "avg_cps": sum(cps_values) / len(cps_values),
        "max_cps": max(cps_values),
        "total_duration": max(c.end for c in cues),
    }


def write_text_file(path: str, content: str) -> str:
    """Write text as UTF-8 (critical for Devanagari and other non-Latin scripts)."""
    with open(path, "w", encoding="utf-8") as handle:
        handle.write(content)
    return path
