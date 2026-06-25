<div align="center">

# 🎬 Video & Audio Transcript System

**Turn any audio or video — in any language — into clean, broadcast-standard subtitles.
Then translate, summarize, and search them, all from your browser.**

[![Python](https://img.shields.io/badge/Python-3.9+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![Gradio](https://img.shields.io/badge/Gradio-UI-F97316?style=for-the-badge&logo=gradio&logoColor=white)](https://www.gradio.app/)
[![Whisper](https://img.shields.io/badge/Whisper-large--v3-412991?style=for-the-badge&logo=openai&logoColor=white)](https://github.com/SYSTRAN/faster-whisper)
[![Groq](https://img.shields.io/badge/Groq-LLM-F55036?style=for-the-badge)](https://groq.com/)
[![Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97%20Spaces-Deploy-FFD21E?style=for-the-badge)](https://huggingface.co/spaces)


</div>

---

## 🎥 Demo

<!-- 📌 TODO: While editing this README on github.com, drag your demo .mp4 into the
     line below — GitHub uploads it and turns it into a playable video automatically.
     (Or paste the uploaded video URL here to replace the placeholder.) -->

> 🎬 **Demo video coming soon.** _(Drop your `.mp4` here.)_

---

## ✨ Features

- **🎙️ Two ways in** — upload an audio/video file **or** record straight from your microphone.
- **🌍 Any language** — automatic language detection (Whisper supports ~99 languages), tuned for Indian languages (Hindi, Bengali, Tamil, Telugu, Marathi, Gujarati, Kannada, Malayalam, Punjabi, Urdu) first.
- **⏱️ Broadcast-standard subtitles, not raw text** — output follows real subtitling rules: ≤42 chars/line, ≤2 lines per cue, reading-speed limits, minimum/maximum on-screen durations, and automatic splitting of long speech.
- **🈯 AI translation** — one click translates the subtitles into another language (powered by **Groq**) and exports a **translated SRT** and a **bilingual (stacked) SRT**.
- **📝 AI summary** — get the **5–7 key points** of the transcript as bullet points, with a responsive loading state.
- **🔎 Instant transcript search** — find any word or phrase (case-insensitive, **works with non-Latin scripts** like Hindi), with matches highlighted in place and labelled with their timestamps. Runs fully in the browser.
- **📺 In-browser preview** — play the video back with the generated captions burned over the timeline.
- **📦 Multiple exports** — `.srt`, `.vtt` (WebVTT), and plain `.txt`, all UTF-8.

---

## 🧱 Tech Stack

| Layer | Technology |
|------|------------|
| **Speech-to-text** | ![Whisper](https://img.shields.io/badge/faster--whisper-Whisper%20large--v3-412991?logo=openai&logoColor=white) |
| **LLM (translate + summarize)** | ![Groq](https://img.shields.io/badge/Groq-Llama%203.3%2070B-F55036) |
| **Web UI** | ![Gradio](https://img.shields.io/badge/Gradio-F97316?logo=gradio&logoColor=white) |
| **Search** | ![JavaScript](https://img.shields.io/badge/Vanilla%20JS-client--side-F7DF1E?logo=javascript&logoColor=black) |
| **Language / Packaging** | ![Python](https://img.shields.io/badge/Python-3776AB?logo=python&logoColor=white) · ![Docker](https://img.shields.io/badge/Docker-2496ED?logo=docker&logoColor=white) |

---

## 🏗️ Architecture

A thin UI over a clear, reusable pipeline:

```
  You (browser)
      │  upload a video / record your mic
      ▼
┌───────────────────────────────────────────────────────────────┐
│  app.py — Gradio UI (upload/record, preview, search, summary)  │
└───────────────┬───────────────────────────────────────────────┘
                │ hands work to…
                ▼
┌───────────────────────────────────────────────────────────────┐
│  src/pipeline.py — orchestrates the steps in order             │
└───┬──────────────────┬──────────────────┬────────────────────┘
    ▼                  ▼                  ▼
transcription.py   translation.py     summary.py
(Whisper STT)      (Groq translate)   (Groq summarize)
    │                  │                  │
    └──────────────────┴──────────────────┘
                ▼
         srt_utils.py — broadcast-standard SRT / VTT / TXT
                ▼
   .srt / .vtt / .txt  +  on-screen transcript + captioned preview

  src/auth.py    → resolves API keys (env or services.json) → LLM client
  src/config.py  → settings, models, subtitle standards, languages
  src/static/transcript_search.js → search runs in the browser
```

**Why these choices**
- **faster-whisper (Whisper large-v3)** is the most robust open multilingual ASR, ~4× faster than the reference model, and decodes audio directly from video containers. VAD is enabled to drop silences for cleaner cue timing.
- **Subtitle quality is treated as a first-class problem** — the pure, fully-tested `srt_utils.py` turns a raw transcript into *usable* subtitles (line length, reading speed, cue splitting, correct SRT comma vs VTT period timestamps).
- **Groq** serves fast, open instruction-tuned models over an OpenAI-compatible API, keeping batched translation and summarization low-latency. The provider is swappable via `LLM_PROVIDER` (Anthropic Claude is supported for backwards compatibility).
- **Clean separation of concerns** — `config` · `auth` · `transcription` · `srt_utils` · `translation` · `summary` · `pipeline` · `app`. The error-prone formatting core is unit-tested with no model or network; the LLM paths are tested with a mocked client.

---

## 🚀 Quickstart

### Prerequisites
- **Python 3.9+**
- **ffmpeg** (to decode audio/video) — macOS: `brew install ffmpeg` · Ubuntu: `sudo apt install ffmpeg` · Windows: download from ffmpeg.org
- A **Groq API key** (free) for translation + summarization — get one at [console.groq.com/keys](https://console.groq.com/keys). *(Transcription and search work without any key.)*

### Install & run
```bash
git clone git@github.com:Ronxak/Video-and-Audio-Transcript-System.git
cd Video-and-Audio-Transcript-System

python -m venv venv && source venv/bin/activate     # Windows: venv\Scripts\activate
pip install -r requirements.txt

cp .env.example .env        # then set GROQ_API_KEY=gsk_... inside .env
python app.py               # open the local URL it prints (http://127.0.0.1:7860)
```

> 💡 On a CPU-only machine, set `WHISPER_MODEL=small` (or `base`) in `.env` for much faster runs — `large-v3` is best on a GPU.

---

## 🎛️ Usage

1. **Upload** an audio/video file **or record** with your microphone.
2. (Optional) pick the spoken language, or leave it on **Auto-detect**.
3. (Optional) tick **Translate transcript** and choose a target language.
4. Click **Generate transcript** — read the preview, watch the captioned video, and download `.srt` / `.vtt` / `.txt` (+ translated & bilingual SRT).
5. Click **Summarize** for a 5–7 bullet summary.
6. Use the **Search** box to find any word/phrase — matches are highlighted with their timestamps.

---

## ⚙️ Configuration

All optional, via environment variables or `.env`:

| Variable | Default | Notes |
|----------|---------|-------|
| `GROQ_API_KEY` | — | required for translation + summarization |
| `LLM_PROVIDER` | `groq` | `groq` or `anthropic` (backwards compat) |
| `GROQ_MODEL` | `llama-3.3-70b-versatile` | `llama-3.1-8b-instant` for speed/cost |
| `WHISPER_MODEL` | `large-v3` | `medium` / `small` / `base` for weaker hardware |
| `DEVICE` | auto | `cuda` or `cpu` |
| `COMPUTE_TYPE` | `float16` (GPU) / `int8` (CPU) | precision/speed trade-off |
| `ANTHROPIC_API_KEY` | — | only when `LLM_PROVIDER=anthropic` |
| `MAX_FILE_SIZE_MB` | `500` | upload guard |

> Keys can also be supplied via a `services.json` file (handy on hosted platforms); the environment takes precedence. See `src/auth.py`.

---

## ✅ Testing

No model, network, or API key required — the LLM paths use a mocked client.

```bash
python -m unittest discover -s tests -p "test_*.py"   # 55 Python tests
node tests/test_search_js.js                          # client-side search (incl. Hindi)
```

Coverage: subtitle timestamps/wrapping/timing/cue-splitting & SRT/VTT output, API-key resolution, both translation paths, summary parsing, the summarize pipeline, and the unicode-safe search highlighter.

---

## 🗂️ Project Structure

```
Video-and-Audio-Transcript-System/
├── app.py                  # Gradio UI (thin layer over the pipeline)
├── src/
│   ├── config.py           # settings, models, subtitle standards, languages
│   ├── auth.py             # API-key resolution + Groq/Anthropic clients
│   ├── transcription.py    # faster-whisper wrapper (+ VAD, progress)
│   ├── srt_utils.py        # PURE subtitle formatting (SRT/VTT/TXT) — tested
│   ├── translation.py      # batched LLM translation (Groq default)
│   ├── summary.py          # AI summarization via Groq
│   ├── pipeline.py         # orchestrates the full flow
│   └── static/
│       └── transcript_search.js   # client-side search/highlight (tested)
├── tests/                  # 55 Python tests + a Node test for search
├── requirements.txt
├── .env.example
├── Dockerfile              # optional CPU build
├── LICENSE
└── README.md
```

---

## ☁️ Deploy your own (Hugging Face Spaces, free)

1. Create a free [Hugging Face](https://huggingface.co/join) account and a **write** [access token](https://huggingface.co/settings/tokens).
2. From this folder: `gradio deploy` → choose **CPU basic (free)**.
3. In the Space **Settings**: add a **secret** `GROQ_API_KEY`, and a **variable** `WHISPER_MODEL=small`.

---

## 📄 License

Released under the [MIT License](LICENSE). © 2026 Ronxak.
