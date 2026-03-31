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
) -> list:
    """Run pyannote speaker diarization on *wav_path*.

    Returns a list of ``{"start": float, "end": float, "speaker": str}`` dicts.
    """
    try:
        from pyannote.audio import Pipeline  # type: ignore
    except ImportError as exc:
        raise ImportError(
            "pyannote.audio is not installed. Run: pip install pyannote.audio"
        ) from exc

    pipeline = Pipeline.from_pretrained(
        "pyannote/speaker-diarization-3.1",
        use_auth_token=hf_token,
    )

    # Prefer MPS on Apple Silicon, fall back to CPU
    try:
        import torch  # type: ignore
        if torch.backends.mps.is_available():
            pipeline.to(torch.device("mps"))
    except ImportError:
        pass

    diarize_kwargs: dict = {}
    if num_speakers is not None:
        diarize_kwargs["num_speakers"] = num_speakers

    diarization = pipeline(str(wav_path), **diarize_kwargs)

    return [
        {"start": turn.start, "end": turn.end, "speaker": speaker}
        for turn, _, speaker in diarization.itertracks(yield_label=True)
    ]


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
    result = mlx_whisper.transcribe(
        str(wav_path),
        path_or_hf_repo=repo,
        language=language,
        verbose=False,
    )

    whisper_segments = result.get("segments", [])

    # --- Step 2: Diarization via pyannote.audio (optional) ---
    diarize_segments: list = []
    if hf_token:
        diarize_segments = _diarize(wav_path, hf_token, num_speakers=num_speakers)

    # --- Step 3: Merge transcription + diarization ---
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
