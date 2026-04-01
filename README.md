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

## Pre-flight Checklist

Complete all of these steps **before** running the tool for the first time.
Skipping any of them will cause the pipeline to fail mid-run, potentially after
spending 30+ minutes on transcription.

### 1. Install ffmpeg

```bash
brew install ffmpeg
```

Verify: `ffmpeg -version`

### 2. Install and start Ollama (if using the default local LLM backend)

1. Download and install Ollama from <https://ollama.com/download>
2. Open the Ollama app (or run `ollama serve` in a terminal) so the server is
   running before you invoke `ttrpg-narrator`
3. Pull the model you intend to use:

```bash
ollama pull llama3        # default model (~4.7 GB)
# or any other model you prefer, e.g.:
ollama pull mistral
```

Verify: `ollama list` should show your model.

> **Note:** `ollama serve` must be running for the duration of Phase 3/4.
> If you restart your Mac or close the Ollama app you will need to start it
> again before re-running.

### 3. Set up HuggingFace access (required for speaker diarization)

Speaker diarization uses three gated pyannote models. You must accept the terms
of use for **all three** on the HuggingFace website or diarization will fail
with an HTTP 403 error — and you will have to repeat the 30-minute Whisper
transcription step unnecessarily.

**Do all of this before your first run.**

#### 3a. Create a HuggingFace account and access token

1. Sign up or log in at <https://huggingface.co>
2. Go to **Settings → Access Tokens** → create a token with at least *read*
   permissions
3. Copy the token — you will pass it as `--hf-token` or set it as `$HF_TOKEN`

#### 3b. Accept the terms for all three gated models

Open each link below while logged in to HuggingFace and click **"Agree and
access repository"**:

| Model | URL |
|-------|-----|
| `pyannote/speaker-diarization-3.1` | <https://huggingface.co/pyannote/speaker-diarization-3.1> |
| `pyannote/segmentation-3.0` | <https://huggingface.co/pyannote/segmentation-3.0> |
| `pyannote/speaker-diarization-community-1` | https://huggingface.co/pyannote/speaker-diarization-community-1 |
| `pyannote/wespeaker-voxceleb-resnet34-LM` | <https://huggingface.co/pyannote/wespeaker-voxceleb-resnet34-LM> |

Access approval is instant. All three are required — pyannote loads them as
sub-dependencies and will raise a separate access error for each one you missed.

#### 3c. Set your token in the environment

Add to your .env file:

```bash
HF_TOKEN="hf_your_token_here"
```

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

> **Before running:** make sure Ollama is running (`ollama serve` or the Ollama
> app is open), you have pulled a model (`ollama pull llama3`), and you have
> accepted all three pyannote model agreements on HuggingFace (see Pre-flight
> Checklist above).

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

