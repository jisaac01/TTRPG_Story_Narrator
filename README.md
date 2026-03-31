# TTRPG Story Narrator

> **"Project ScrybeClone"** — Turn a folder of TTRPG session recordings into a
> polished, third-person narrative prose summary.

---

## Overview

`ttrpg-narrator` is a Python CLI tool that processes multiple TTRPG audio
recordings through four automated phases:

| Phase | Name | What it does |
|-------|------|--------------|
| 1 | **The Joiner** | Sorts `.m4a` files by *Media Created* metadata, concatenates them with `ffmpeg`, and converts to a 16 kHz mono WAV. |
| 2 | **The Ear** | Transcribes the WAV with [mlx-whisper](https://github.com/ml-explore/mlx-examples/tree/main/whisper) (Apple Silicon native) and diarizes speakers with [pyannote.audio](https://github.com/pyannote/pyannote-audio), producing a structured JSON transcript. |
| 3 | **The Writer** | Uses an LLM to (A) strip out-of-character "table talk" and (B) rewrite the remaining dialogue/actions as narrative prose. |
| 4 | **Output** | Saves the finished story as a `.md` file. |

---

## Requirements

- Python ≥ 3.11
- Apple Silicon Mac (M1/M2/M3/M4) — mlx-whisper requires Apple MLX
- [`ffmpeg`](https://ffmpeg.org/) installed and on `$PATH`
- A HuggingFace access token (for speaker diarization via pyannote) — optional
  but recommended
- One of:
  - A running [Ollama](https://ollama.ai/) server for local LLM inference
  - A Google [Gemini API key](https://aistudio.google.com/app/apikey)

---

## Installation

> **Always work inside a virtual environment** — never install packages into
> your system or user Python.

```bash
# 1. Create and activate a virtual environment (requires Python 3.11+)
/opt/homebrew/bin/python3.11 -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate

# 2a. Install the package in editable mode (recommended for development)
pip install -e .

# 2b. — OR — install from the pinned lockfile (recommended for reproducibility)
pip install -r requirements.lock.txt
```

To deactivate the virtual environment when you're done:

```bash
deactivate
```

---

## Quick Start

### Full pipeline (recommended)

```bash
ttrpg-narrator narrate ./recordings/
```

This will:
1. Find all `.m4a` files in `./recordings/`, sort them by creation timestamp,
   join and convert them to a WAV.
2. Transcribe the WAV with mlx-whisper and diarize speakers with pyannote.audio
   (speaker labels require `--hf-token`).
3. Clean table talk and generate narrative prose with Ollama (`llama3` by
   default).
4. Save the story to `./recordings/story.md`.

### With all options

```bash
# Export your Gemini API key before running — do NOT pass it as a flag.
export GEMINI_API_KEY="your-key-here"

ttrpg-narrator narrate ./recordings/ \
  --output ./my_session_story.md \
  --backend gemini \
  --model gemini-1.5-pro \
  --hf-token "$HF_TOKEN" \
  --whisper-model large-v3 \
  --num-speakers 4 \
  --keep-work
```

---

## Commands

### `narrate`  — full pipeline

```
ttrpg-narrator narrate [OPTIONS] INPUT_FOLDER
```

| Option | Default | Description |
|--------|---------|-------------|
| `--output / -o` | `<input_folder>/story.md` | Output Markdown file |
| `--work-dir / -w` | `<input_folder>/.narrator_work` | Intermediate file directory |
| `--backend / -b` | `ollama` | LLM backend: `ollama` or `gemini` |
| `--model / -m` | `llama3` / `gemini-1.5-pro` | Model name for the chosen backend |
| `--hf-token` | `$HF_TOKEN` | HuggingFace token for pyannote diarization |
| `--whisper-model` | `large-v3` | mlx-whisper model name or HuggingFace repo ID |
| `--language` | `en` | Language code (skip auto-detection) |
| `--num-speakers` | auto | Exact speaker count hint for diarization |
| `--skip-join` | off | Skip Phase 1 (reuse existing WAV) |
| `--skip-transcribe` | off | Skip Phase 2 (reuse existing JSON) |
| `--keep-work` | off | Keep intermediate files after completion |

### `join`  — Phase 1 only

```bash
ttrpg-narrator join ./recordings/
```

Produces `full_session.wav` (and `full_session.m4a`) in
`./recordings/.narrator_work/`.

### `transcribe`  — Phase 2 only

```bash
ttrpg-narrator transcribe ./recordings/.narrator_work/full_session.wav \
  --hf-token "$HF_TOKEN"
```

Produces a `full_session.json` transcript next to the WAV.

### `write`  — Phase 3 + 4 only

```bash
ttrpg-narrator write ./transcript.json \
  --backend ollama --model llama3 \
  --output ./story.md
```

---

## Transcript JSON Format

The Phase 2 output (and Phase 3 input) is a JSON array of segment objects:

```json
[
  {"start": 1.2,  "end": 4.5,  "speaker": "SPEAKER_00", "text": "I roll for initiative."},
  {"start": 5.0,  "end": 8.3,  "speaker": "SPEAKER_01", "text": "The goblin lunges forward!"},
  ...
]
```

---

## Environment Variables

| Variable | Purpose |
|----------|----------|
| `GEMINI_API_KEY` | Gemini API key — **required** when `--backend gemini`; read from the environment only (never passed as a CLI flag) |
| `HF_TOKEN` | HuggingFace token for pyannote speaker diarization |

---

## Project Structure

```
ttrpg_narrator/
├── __init__.py           # Package version
├── cli.py                # Click CLI entry-point
├── joiner.py             # Phase 1: audio pre-processing
├── transcriber.py        # Phase 2: mlx-whisper transcription + pyannote diarization
└── writer.py             # Phase 3+4: LLM narrative synthesis + Markdown output
pyproject.toml
requirements.txt          # Abstract dependencies (lower-bound pins)
requirements.lock.txt     # Fully pinned + hashed lockfile (generated by pip-compile)
```

