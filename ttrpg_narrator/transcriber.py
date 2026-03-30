"""Phase 2 — Transcription & Diarization (The "Ear").

Uses WhisperX to transcribe a WAV file, align word-level timestamps,
and assign speaker labels via pyannote diarization.

Output is a list of dicts with the shape::

    {"start": 1.2, "end": 4.5, "speaker": "SPEAKER_00", "text": "I roll for initiative."}
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import List, Optional


# ---------------------------------------------------------------------------
# Types
# ---------------------------------------------------------------------------

Segment = dict  # {"start": float, "end": float, "speaker": str, "text": str}


# ---------------------------------------------------------------------------
# WhisperX pipeline
# ---------------------------------------------------------------------------

def transcribe(
    wav_path: Path,
    *,
    hf_token: Optional[str] = None,
    model_name: str = "large-v2",
    device: str = "cpu",
    compute_type: str = "int8",
    batch_size: int = 8,
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
        WhisperX / Whisper model size (e.g. ``"large-v2"``, ``"medium"``).
    device:
        PyTorch device string (``"cpu"``, ``"cuda"``, ``"mps"``).
    compute_type:
        Quantization level (``"int8"``, ``"float16"``, ``"float32"``).
    batch_size:
        Batch size for the WhisperX inference loop.
    num_speakers:
        Optional hint for pyannote — the exact number of speakers present.

    Returns
    -------
    list of dict
        Each dict: ``{"start": float, "end": float, "speaker": str, "text": str}``.
    """
    try:
        import whisperx  # type: ignore
    except ImportError as exc:
        raise ImportError(
            "whisperx is not installed. Run: pip install whisperx"
        ) from exc

    audio = whisperx.load_audio(str(wav_path))

    # --- Step 1: Transcription ---
    model = whisperx.load_model(
        model_name, device, compute_type=compute_type
    )
    result = model.transcribe(audio, batch_size=batch_size)
    language = result.get("language", "en")

    # --- Step 2: Alignment ---
    align_model, metadata = whisperx.load_align_model(
        language_code=language, device=device
    )
    result = whisperx.align(
        result["segments"], align_model, metadata, audio, device,
        return_char_alignments=False,
    )

    # --- Step 3: Diarization (optional) ---
    if hf_token:
        diarize_model = whisperx.DiarizationPipeline(
            use_auth_token=hf_token, device=device
        )
        diarize_kwargs: dict = {}
        if num_speakers is not None:
            diarize_kwargs["num_speakers"] = num_speakers
        diarize_segments = diarize_model(audio, **diarize_kwargs)
        result = whisperx.assign_word_speakers(diarize_segments, result)

    # --- Step 4: Flatten to output format ---
    segments: List[Segment] = []
    for seg in result.get("segments", []):
        speaker = seg.get("speaker", "SPEAKER_00")
        text = seg.get("text", "").strip()
        if not text:
            continue
        segments.append(
            {
                "start": round(float(seg.get("start", 0.0)), 3),
                "end": round(float(seg.get("end", 0.0)), 3),
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
