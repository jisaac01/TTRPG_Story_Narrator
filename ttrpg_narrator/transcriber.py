"""Phase 2 — Transcription & Diarization (The "Ear").

Uses mlx-whisper (Apple Silicon native) to transcribe a WAV file and
pyannote.audio to assign speaker labels, producing a structured JSON transcript.

Output is a list of dicts with the shape::

    {"start": 1.2, "end": 4.5, "speaker": "SPEAKER_00", "text": "I roll for initiative."}
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import List, Optional

Segment = dict  # {"start": float, "end": float, "speaker": str, "text": str}


# ---------------------------------------------------------------------------
# Speaker assignment helper
# ---------------------------------------------------------------------------

def _assign_speaker(start: float, end: float, diarize_segments: list) -> str:
    """Return the speaker label with the greatest overlap for [start, end]."""
    max_overlap = 0.0
    best_speaker = "SPEAKER_00"
    for dseg in diarize_segments:
        dstart = dseg["start"]
        dend = dseg["end"]
        overlap = min(end, dend) - max(start, dstart)
        if overlap > max_overlap:
            max_overlap = overlap
            best_speaker = dseg["speaker"]
    return best_speaker


# ---------------------------------------------------------------------------
# pyannote.audio diarization
# ---------------------------------------------------------------------------

def _diarize(
    wav_path: Path,
    hf_token: str,
    num_speakers: Optional[int] = None,
    log=None,
    diarize_cache_path: Optional[Path] = None,
) -> list:
    """Run pyannote speaker diarization on *wav_path*.

    Returns a list of ``{"start": float, "end": float, "speaker": str}`` dicts.
    """
    if log is None:
        log = print

    # --- Load from cache if available ---
    if diarize_cache_path is not None and diarize_cache_path.exists():
        log(f"      Loading cached diarization from {diarize_cache_path}…")
        with open(diarize_cache_path, encoding="utf-8") as _fh:
            turns = json.load(_fh)
        unique_speakers = len({t["speaker"] for t in turns})
        log(f"      Loaded {len(turns)} cached turns, {unique_speakers} speaker(s).")
        return turns

    try:
        from pyannote.audio import Pipeline  # type: ignore
    except ImportError as exc:
        raise ImportError(
            "pyannote.audio is not installed. Run: pip install pyannote.audio"
        ) from exc

    log("      Loading pyannote diarization pipeline…")
    pipeline = Pipeline.from_pretrained(
        "pyannote/speaker-diarization-3.1",
        token=hf_token,
    )

    # Prefer MPS on Apple Silicon, fall back to CPU
    try:
        import torch  # type: ignore
        if torch.backends.mps.is_available():
            pipeline.to(torch.device("mps"))
            log("      Using MPS (Apple Silicon) for diarization.")
        else:
            log("      Using CPU for diarization.")
    except ImportError:
        log("      Using CPU for diarization (torch not available).")

    diarize_kwargs: dict = {}
    if num_speakers is not None:
        diarize_kwargs["num_speakers"] = num_speakers
        log(f"      Running diarization (num_speakers={num_speakers})…")
    else:
        log("      Running diarization (auto-detecting speaker count)…")

    # Hook pyannote's internal progress reporting into our log callback.
    def _on_progress(sender, completed, total, **kwargs):
        if total:
            pct = int(100 * completed / total)
            log(f"      [2b] Diarization progress: {completed}/{total} ({pct}%)")

    pipeline.progress_hook = _on_progress

    diarization = pipeline(str(wav_path), **diarize_kwargs)

    # pyannote.audio 4.x returns a DiarizeOutput dataclass; 3.x returns
    # an Annotation directly.  Normalise both to an Annotation.
    annotation = (
        diarization.speaker_diarization
        if hasattr(diarization, "speaker_diarization")
        else diarization
    )

    turns = [
        {"start": round(turn.start, 3), "end": round(turn.end, 3), "speaker": speaker}
        for turn, _, speaker in annotation.itertracks(yield_label=True)
    ]
    unique_speakers = len({t["speaker"] for t in turns})
    log(f"      Diarization complete — {len(turns)} turns, {unique_speakers} speaker(s) detected.")

    # Save checkpoint immediately so a later crash doesn't lose this work.
    if diarize_cache_path is not None:
        diarize_cache_path.parent.mkdir(parents=True, exist_ok=True)
        with open(diarize_cache_path, "w", encoding="utf-8") as _fh:
            json.dump(turns, _fh)
        log(f"      Diarization cached → {diarize_cache_path}")

    return turns


# ---------------------------------------------------------------------------
# Short-name → full HuggingFace repo ID mapping for mlx-community models
# ---------------------------------------------------------------------------

#: Maps convenient short names to their ``mlx-community`` HuggingFace repo IDs.
#: If ``model_name`` already contains a ``/`` it is treated as a full repo ID
#: (or local path) and this table is not consulted.
_MLX_MODEL_REPOS: dict[str, str] = {
    "tiny":      "mlx-community/whisper-tiny-mlx",
    "tiny.en":   "mlx-community/whisper-tiny.en-mlx",
    "base":      "mlx-community/whisper-base-mlx",
    "base.en":   "mlx-community/whisper-base.en-mlx",
    "small":     "mlx-community/whisper-small-mlx",
    "small.en":  "mlx-community/whisper-small.en-mlx",
    "medium":    "mlx-community/whisper-medium-mlx",
    "medium.en": "mlx-community/whisper-medium.en-mlx",
    "large":     "mlx-community/whisper-large-mlx",
    "large-v1":  "mlx-community/whisper-large-v1-mlx",
    "large-v2":  "mlx-community/whisper-large-v2-mlx",
    "large-v3":  "mlx-community/whisper-large-v3-mlx",
    "turbo":     "mlx-community/whisper-large-v3-turbo",
}


def _resolve_model(model_name: str) -> str:
    """Return the full HF repo ID (or local path) for *model_name*.

    If *model_name* already contains a ``/`` it is returned unchanged,
    allowing callers to pass a full repo ID such as
    ``"mlx-community/whisper-large-v3-mlx"`` or a local path.
    Otherwise the name is looked up in :data:`_MLX_MODEL_REPOS`; if not found
    it is returned as-is so mlx-whisper can produce its own error.
    """
    if "/" in model_name:
        return model_name
    return _MLX_MODEL_REPOS.get(model_name, model_name)


# ---------------------------------------------------------------------------
# mlx-whisper transcription + diarization pipeline
# ---------------------------------------------------------------------------

def transcribe(
    wav_path: Path,
    *,
    hf_token: Optional[str] = None,
    model_name: str = "large-v3",
    language: str = "en",
    num_speakers: Optional[int] = None,
    log=None,
    whisper_cache_path: Optional[Path] = None,
    diarize_cache_path: Optional[Path] = None,
) -> List[Segment]:
    """Transcribe *wav_path* and return diarized segments.

    Parameters
    ----------
    wav_path:
        Path to the 16 kHz mono WAV produced by :mod:`ttrpg_narrator.joiner`.
    hf_token:
        HuggingFace access token required by the pyannote diarization models.
        If *None* the diarization step is skipped and speaker labels default
        to ``"SPEAKER_00"``.
    model_name:
        mlx-whisper model name or HuggingFace repo ID.  Short names such as
        ``"large-v3"``, ``"large-v2"``, ``"medium"``, and ``"turbo"`` are
        resolved automatically to the corresponding ``mlx-community`` repo.
    language:
        Language code (e.g. ``"en"``) to skip auto-detection and speed up
        processing.
    num_speakers:
        Optional hint for pyannote — the exact number of speakers present.

    Returns
    -------
    list of dict
        Each dict: ``{"start": float, "end": float, "speaker": str, "text": str}``.
    """
    if log is None:
        log = print

    try:
        import mlx_whisper  # type: ignore
    except ImportError as exc:
        raise ImportError(
            "mlx-whisper is not installed. Run: pip install mlx-whisper"
        ) from exc

    # Resolve short names (e.g. "large-v3") to full mlx-community repo IDs.
    # mlx-whisper (and the HuggingFace hub) cache downloaded weights under
    # ~/.cache/huggingface/hub/, so subsequent runs skip the download entirely.
    repo = _resolve_model(model_name)

    # --- Step 1: Transcription via mlx-whisper ---
    if whisper_cache_path is not None and whisper_cache_path.exists():
        log(f"      [2a] Loading cached Whisper segments from {whisper_cache_path}…")
        with open(whisper_cache_path, encoding="utf-8") as _fh:
            whisper_segments = json.load(_fh)
        log(f"      [2a] Loaded {len(whisper_segments)} cached segments (skipping Whisper).")
    else:
        log(f"      [2a] Starting Whisper transcription (model={repo}, language={language})…")
        result = mlx_whisper.transcribe(
            str(wav_path),
            path_or_hf_repo=repo,
            language=language,
            verbose=False,
        )
        whisper_segments = result.get("segments", [])
        log(f"      [2a] Whisper done — {len(whisper_segments)} raw segments.")
        if whisper_cache_path is not None:
            whisper_cache_path.parent.mkdir(parents=True, exist_ok=True)
            with open(whisper_cache_path, "w", encoding="utf-8") as _fh:
                json.dump(whisper_segments, _fh)
            log(f"      [2a] Whisper segments cached → {whisper_cache_path}")

    # --- Step 2: Diarization via pyannote.audio (optional) ---
    diarize_segments: list = []
    if hf_token:
        log("      [2b] Starting speaker diarization…")
        diarize_segments = _diarize(
            wav_path,
            hf_token,
            num_speakers=num_speakers,
            log=log,
            diarize_cache_path=diarize_cache_path,
        )
    else:
        log("      [2b] Skipping diarization (no HF token) — all segments labelled SPEAKER_00.")

    # --- Step 3: Merge transcription + diarization ---
    log("      [2c] Merging transcription and speaker labels…")
    segments: List[Segment] = []
    for seg in whisper_segments:
        start = seg.get("start", 0.0)
        end = seg.get("end", 0.0)
        text = seg.get("text", "").strip()
        if not text:
            continue
        speaker = _assign_speaker(float(start), float(end), diarize_segments)
        segments.append(
            {
                "start": round(float(start), 3),
                "end": round(float(end), 3),
                "speaker": speaker,
                "text": text,
            }
        )
    log(f"      [2c] Merge done — {len(segments)} segments with speaker labels.")

    return segments


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------

def save_transcript(segments: List[Segment], output_path: Path) -> Path:
    """Write *segments* as a pretty-printed JSON file."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as fh:
        json.dump(segments, fh, indent=2, ensure_ascii=False)
    return output_path


def load_transcript(json_path: Path) -> List[Segment]:
    """Load a previously saved transcript JSON."""
    with open(json_path, encoding="utf-8") as fh:
        return json.load(fh)
