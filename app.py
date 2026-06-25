"""
Gradio UI for the Video & Audio Transcript System.

Record or upload audio/video in any language, get a broadcast-standard SRT with
timestamps, plus optional AI translation (Groq) for localization-ready
subtitles, an AI summary of the transcript, and in-browser transcript search.

Run:  python app.py
"""
from __future__ import annotations

import base64
import html as html_lib
from pathlib import Path
from urllib.parse import quote

import gradio as gr

from src import config
from src.pipeline import process
from src.srt_utils import cues_to_text
from src.summary import summarize_transcript

LANGUAGE_CHOICES = list(config.SUPPORTED_LANGUAGES.keys())

# --------------------------------------------------------------------------- #
# Transcript rendering (HTML so we can highlight search matches with <mark>)
# --------------------------------------------------------------------------- #
_EMPTY_TRANSCRIPT_HTML = (
    '<div id="ts-search-msg" class="ts-msg" style="display:none"></div>'
    '<div id="ts-transcript" class="transcript-box">'
    '<span class="ts-empty">Your transcript will appear here after you click '
    "“Generate transcript”.</span></div>"
)

# Always-visible "Preview with captions" box states. The box stays on screen so
# it never disappears and Gradio's loading overlay can render inside it while
# processing; the actual player is built by _video_preview_html().
_EMPTY_PREVIEW_HTML = (
    '<div class="video-preview-label">Preview with captions</div>'
    '<div class="video-preview-placeholder">Upload a video and click '
    "<b>Generate transcript</b> to preview it here with captions.</div>"
)
_AUDIO_PREVIEW_HTML = (
    '<div class="video-preview-label">Preview with captions</div>'
    '<div class="video-preview-placeholder">Audio input — no video to preview. '
    "Your transcript and downloads are below.</div>"
)


def build_transcript_html(cues) -> str:
    """Render cues as an HTML block the search JS can highlight in place.

    Each cue carries its raw text and start/end times as data-attributes so the
    client-side search can wrap matches in <mark> and show a timestamp label
    without another server round-trip.
    """
    if not cues:
        return _EMPTY_TRANSCRIPT_HTML

    rows = []
    for cue in cues:
        body = " ".join(cue.text.split())
        mm, ss = divmod(int(cue.start), 60)
        stamp = f"[{mm:02d}:{ss:02d}]"
        rows.append(
            f'<div class="ts-cue" data-start="{cue.start:.3f}" '
            f'data-end="{cue.end:.3f}" data-text="{html_lib.escape(body, quote=True)}">'
            f'<span class="ts-time">{stamp}</span>'
            f'<span class="ts-text">{html_lib.escape(body)}</span></div>'
        )
    return (
        '<div id="ts-search-msg" class="ts-msg" style="display:none"></div>'
        '<div id="ts-transcript" class="transcript-box">' + "".join(rows) + "</div>"
    )


def _video_preview_html(video_path: str, vtt_path: str, lang: str = "und") -> str:
    """Render an HTML5 <video> with a caption <track> shown by default.

    Gradio 6's Video component can't accept subtitles through its value (its
    postprocess only handles a plain path and raises on a (video, subtitle)
    tuple), so we build the player ourselves: the video streams from Gradio's
    file endpoint and the WebVTT is embedded as a base64 data URI so it always
    loads without extra file-serving permissions. The ``default`` attribute turns
    captions on automatically, synced to the video timeline.
    """
    video_url = "/gradio_api/file=" + quote(str(video_path), safe="/")
    with open(vtt_path, encoding="utf-8") as handle:
        vtt_b64 = base64.b64encode(handle.read().encode("utf-8")).decode("ascii")
    srclang = lang if (lang and lang.isalpha() and len(lang) <= 3) else "und"
    return (
        '<div class="video-preview-label">Preview with captions</div>'
        '<video controls crossorigin="anonymous" preload="metadata" '
        'style="width:100%;max-height:440px;border-radius:8px;background:#000">'
        f'<source src="{html_lib.escape(video_url)}">'
        f'<track default kind="subtitles" srclang="{srclang}" label="Captions" '
        f'src="data:text/vtt;base64,{vtt_b64}">'
        "Your browser does not support embedded video."
        "</video>"
    )


# --------------------------------------------------------------------------- #
# Handlers
# --------------------------------------------------------------------------- #
def run(upload, recording, language_label, do_translate, target_language,
        progress=gr.Progress()):
    """Handle a single transcription request from the UI."""
    media = upload or recording
    if not media:
        raise gr.Error("Please upload a file or record some audio first.")

    language = config.SUPPORTED_LANGUAGES.get(language_label)
    translate_to = target_language if do_translate else None

    try:
        result = process(
            media,
            language=language,
            translate_to=translate_to,
            progress=lambda frac, msg="Working…": progress(frac, desc=msg),
        )
    except Exception as exc:  # surface a clean message in the UI
        raise gr.Error(str(exc))

    stats = result.stats
    info_md = (
        f"**Detected language:** `{result.detected_language}`  \n"
        f"**Duration:** {result.duration:.1f}s &nbsp;•&nbsp; "
        f"**Subtitles:** {stats['count']}  \n"
        f"**Avg reading speed:** {stats['avg_cps']:.1f} CPS "
        f"(target ≤ {config.SUBTITLE_STYLE.max_cps:.0f})  \n"
        f"**Engine:** faster-whisper `{config.WHISPER_MODEL}` on `{config.DEVICE}`"
    )
    if result.target_language:
        info_md += f"  \n**Translated to:** {result.target_language}"

    return (
        gr.update(value=info_md, visible=True),
        # Gradio 6's Video can't take a (video, subtitle) value, so render our own
        # HTML5 player with a caption track (see _video_preview_html). The box
        # stays visible with a placeholder for audio so it never disappears.
        gr.update(
            value=_video_preview_html(media, result.vtt_path, result.detected_language)
            if result.is_video else _AUDIO_PREVIEW_HTML,
            visible=True,
        ),
        build_transcript_html(result.cues),
        result.srt_path,
        result.vtt_path,
        result.txt_path,
        result.translated_srt_path,
        result.bilingual_srt_path,
        cues_to_text(result.cues),                 # full plain text -> state
        gr.update(interactive=True),               # enable Summarize
        gr.update(value="", visible=False),        # clear summary status
        gr.update(value="", visible=False),        # clear summary bullets
        gr.update(value=""),                       # reset search box
        gr.update(visible=True),                   # show download section
    )


def _summarize_loading():
    """Immediate loading state shown before the (slow) Groq summary call."""
    return (
        gr.update(interactive=False),
        gr.update(value="Summarizing… this can take a few seconds.", visible=True),
        gr.update(visible=False),
    )


def do_summarize(transcript_text):
    """Summarise the already-generated transcript (Groq) and return the result.

    Paired with ``_summarize_loading`` via ``.click(...).then(...)`` rather than
    written as a generator: a generator that toggles ``visible`` over Gradio's
    queue can drop its final render on the first click (the "works on the second
    click" bug). Two plain functions render reliably every time.
    """
    if not (transcript_text or "").strip():
        return (
            gr.update(interactive=True),
            gr.update(value="Generate a transcript first.", visible=True),
            gr.update(visible=False),
        )

    try:
        bullets = summarize_transcript(transcript_text)
    except Exception as exc:  # SummaryError (missing key) or API failure
        return (
            gr.update(interactive=True),
            gr.update(value=f"⚠️ {exc}", visible=True),
            gr.update(visible=False),
        )

    body = "\n".join(f"- {b}" for b in bullets) if bullets else "_No summary produced._"
    return (
        gr.update(interactive=True),
        gr.update(visible=False),
        gr.update(value="### Summary\n" + body, visible=True),
    )


# --------------------------------------------------------------------------- #
# Client-side transcript search helpers, loaded from a static file so the exact
# same code is unit-tested in tests/test_search_js.js (single source of truth).
# --------------------------------------------------------------------------- #
_SEARCH_JS = (
    Path(__file__).resolve().parent / "src" / "static" / "transcript_search.js"
).read_text(encoding="utf-8")

_CSS = """
/* Center header & footer */
.app-header { text-align: center; }
.app-footer { text-align: center; margin-top: 8px; }
.app-header h1 { margin-bottom: 4px; }
.app-header p {
  color: var(--body-text-color-subdued, #6b7280);
  max-width: 800px; margin: 0 auto 16px;
  text-wrap: balance;
}
/* Hide Gradio progress overlay everywhere … */
.gradio-container .generating { background: transparent !important; border: none !important; }
.gradio-container .progress-level { display: none !important; }
/* … except inside the video preview box */
#video-preview .generating { background: var(--block-background-fill) !important; }
/* Gradio anchors the loading overlay (.wrap.center) to the top of the box; make
   it span the full height so its built-in justify-content:center puts the
   progress bar in the middle, where the placeholder text sits. */
#video-preview .wrap { inset: 0 !important; }
#video-preview .progress-level { display: flex !important; }
/* Preview box: custom HTML5 player + placeholder states */
#video-preview { min-height: 120px; }
.video-preview-label { font-weight: 600; margin-bottom: 6px; }
.video-preview-placeholder {
  display: flex; align-items: center; justify-content: center; text-align: center;
  min-height: 200px; padding: 16px;
  border: 1px dashed var(--border-color-primary, #d1d5db); border-radius: 8px;
  color: var(--body-text-color-subdued, #6b7280);
  background: var(--background-fill-secondary, #f9fafb);
}
/* Transcript box */
.transcript-box {
  max-height: 360px; overflow-y: auto; padding: 12px;
  border: 1px solid var(--border-color-primary, #d1d5db); border-radius: 8px;
  background: var(--background-fill-primary, #fff);
  font-size: 0.95rem; line-height: 1.55;
}
.ts-cue { padding: 1px 0; }
.ts-time { color: #6b7280; font-variant-numeric: tabular-nums; margin-right: 6px; user-select: none; }
.transcript-box mark { background: #fde68a; color: inherit; padding: 0 1px; border-radius: 2px; }
.ts-match { font-size: 0.7em; color: #2563eb; margin-left: 2px; white-space: nowrap; }
.ts-msg { margin-bottom: 8px; font-size: 0.85rem; font-weight: 600; color: #b45309; }
.ts-empty { color: #9ca3af; }
"""


with gr.Blocks(
    title="Video & Audio Transcript System",
) as demo:
    gr.Markdown(
        "# Video & Audio Transcript System\n"
        "Upload **or record** audio/video in any language (e.g. **Hindi**) and get a "
        "clean, broadcast-standard **SRT** with timestamps — with "
        "**AI translation**, an **AI summary**, and instant **transcript search**.",
        elem_classes=["app-header"],
    )

    # Holds the full plain transcript text for the Summarize step.
    transcript_state = gr.State("")

    with gr.Row():
        with gr.Column(scale=1):
            upload = gr.File(
                label="Upload audio or video",
                file_types=["audio", "video"],
                type="filepath",
            )
            recording = gr.Audio(
                label="Record audio",
                sources=["microphone"],
                type="filepath",
            )
            language = gr.Dropdown(
                LANGUAGE_CHOICES, value="Auto-detect", label="Spoken language"
            )
            with gr.Group():
                do_translate = gr.Checkbox(
                    label="Translate transcript", value=False
                )
                target = gr.Dropdown(
                    config.TARGET_LANGUAGES, value="English", label="Translate to"
                )
            with gr.Row():
                run_btn = gr.Button(
                    "Generate transcript", variant="primary", size="lg"
                )
                summarize_btn = gr.Button(
                    "Summarize", variant="secondary", size="lg", interactive=False
                )

        with gr.Column(scale=2):
            info = gr.Markdown(visible=False)
            video_preview = gr.HTML(
                value=_EMPTY_PREVIEW_HTML, elem_id="video-preview"
            )

            with gr.Column() as transcript_section:
                gr.Markdown("#### Transcript")
                with gr.Row():
                    search_box = gr.Textbox(
                        label="Search transcript",
                        placeholder="Find a word or phrase…  (Enter to search)",
                        scale=4,
                    )
                    search_btn = gr.Button("Search", scale=1)
                transcript_view = gr.HTML(value=_EMPTY_TRANSCRIPT_HTML)

            summary_status = gr.Markdown(visible=False)
            summary_output = gr.Markdown(visible=False)

            with gr.Column(visible=False) as download_section:
                gr.Markdown("#### Downloads")
                with gr.Row():
                    srt_file = gr.File(label="SRT")
                    vtt_file = gr.File(label="VTT")
                    txt_file = gr.File(label="Plain text")
                with gr.Row():
                    translated_file = gr.File(label="Translated SRT")
                    bilingual_file = gr.File(label="Bilingual SRT")

    run_btn.click(
        run,
        inputs=[upload, recording, language, do_translate, target],
        outputs=[
            info,
            video_preview,
            transcript_view,
            srt_file,
            vtt_file,
            txt_file,
            translated_file,
            bilingual_file,
            transcript_state,
            summarize_btn,
            summary_status,
            summary_output,
            search_box,
            download_section,
        ],
    )

    summarize_btn.click(
        _summarize_loading,
        inputs=None,
        outputs=[summarize_btn, summary_status, summary_output],
    ).then(
        do_summarize,
        inputs=[transcript_state],
        outputs=[summarize_btn, summary_status, summary_output],
    )

    # Search runs entirely client-side (no server round-trip → instant).
    search_btn.click(None, inputs=[search_box], outputs=None,
                     js="(q) => window.__tsSearch(q)")
    search_box.submit(None, inputs=[search_box], outputs=None,
                      js="(q) => window.__tsSearch(q)")
    # Clear highlights as soon as the query changes.
    search_box.change(None, inputs=[search_box], outputs=None,
                      js="() => window.__tsClear()")

    # Register the search helpers on the window once the page loads.
    demo.load(None, None, None, js=_SEARCH_JS)

    gr.Markdown(
        "<sub>Built with faster-whisper (Whisper large-v3) for transcription and "
        "Groq for translation & summarization. Output follows broadcast subtitle "
        "standards (≤42 chars/line, ≤2 lines, reading-speed limits).</sub>",
        elem_classes=["app-footer"],
    )


if __name__ == "__main__":
    # Queue is required for the streaming (generator) Summarize handler.
    demo.queue()
    # share=False keeps it local; set GRADIO_SERVER_NAME=0.0.0.0 to expose in Docker.
    demo.launch(
        theme=gr.themes.Soft(primary_hue="indigo", secondary_hue="slate"),
        css=_CSS,
    )
