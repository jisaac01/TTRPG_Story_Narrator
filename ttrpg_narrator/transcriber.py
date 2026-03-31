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
# Helper for IFW diarization
# ---------------------------------------------------------------------------

def assign_speakers_to_chunks(chunks, diarize_segments):
    """Assign speakers to chunks based on diarization segments."""
    for chunk in chunks:
        start, end = chunk["timestamp"]
        # Find the diarization segment that overlaps most with this chunk
        max_overlap = 0
        best_speaker = "SPEAKER_00"
        for dseg in diarize_segments:
            dstart = dseg["start"]
            dend = dseg["end"]
            overlap = min(end, dend) - max(start, dstart)
            if overlap > max_overlap:
                max_overlap = overlap
                best_speaker = dseg["speaker"]
        chunk["speaker"] = best_speaker
    return chunks

Segment = dict  # {"start": float, "end": float, "speaker": str, "text": str}


# ---------------------------------------------------------------------------
# WhisperX pipeline
# ---------------------------------------------------------------------------

def transcribe(
    wav_path: Path,
    *,
    transcriber: str = "ifw",
    hf_token: Optional[str] = None,
    model_name: str = "large-v3",
    device: str = "mps",
    language: str = "en",
    compute_type: str = "float16",
    batch_size: int = 8,
    num_speakers: Optional[int] = None,
) -> List[Segment]:
    """Transcribe *wav_path* and return diarized segments.

    Parameters
    ----------
    wav_path:
        Path to the 16 kHz mono WAV produced by :mod:`ttrpg_narrator.joiner`.
    transcriber:
        Transcription backend ('whisperx' for accuracy with alignment, 'ifw' for speed).
    hf_token:
        HuggingFace access token required by the pyannote diarization models.
        If *None* the diarization step is skipped and speaker labels default
        to ``"SPEAKER_00"``.
    model_name:
        WhisperX / Whisper model size (e.g. ``"large-v3"``, ``"large-v2"``, ``"medium"``).
    device:
        PyTorch device string (``"cpu"``, ``"cuda"``, ``"mps"``).
    language:
        Language code (e.g. ``"en"``) to skip auto-detection and speed up processing.
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

    if transcriber == "ifw":
        try:
            from transformers import pipeline
            import torch
        except ImportError as exc:
            raise ImportError(
                "transformers is not installed. Run: pip install transformers torch"
            ) from exc

    audio = whisperx.load_audio(str(wav_path))

    if transcriber == "whisperx":
        # --- Step 1: Transcription ---
        model = whisperx.load_model(
            model_name, device, compute_type=compute_type
        )
        result = model.transcribe(audio, language=language, batch_size=batch_size)

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

    elif transcriber == "ifw":
        # Adjust model_name for HuggingFace
        if "/" not in model_name:
            model_name = f"openai/whisper-{model_name}"
        # Use Insanely Fast Whisper via transformers pipeline
        pipe = pipeline(
            "automatic-speech-recognition",
            model=model_name,
            torch_dtype=torch.float16,
            device=device,
            model_kwargs={},
        )
        generate_kwargs = {"task": "transcribe", "language": language}
        result = pipe(audio, generate_kwargs=generate_kwargs, return_timestamps=True)
        # Result has 'chunks' with segments

        # --- Step 3: Diarization (optional) ---
        if hf_token:
            diarize_model = whisperx.DiarizationPipeline(
                use_auth_token=hf_token, device=device
            )
            diarize_kwargs: dict = {}
            if num_speakers is not None:
                diarize_kwargs["num_speakers"] = num_speakers
            diarize_segments = diarize_model(audio, **diarize_kwargs)
            # Assign speakers to chunks
            result["chunks"] = assign_speakers_to_chunks(result["chunks"], diarize_segments)

    # --- Step 4: Flatten to output format ---
    segments: List[Segment] = []
    seg_list = result.get("segments") or result.get("chunks", [])
    for seg in seg_list:
        if transcriber == "ifw":
            start, end = seg["timestamp"]
            text = seg["text"]
            speaker = "SPEAKER_00"
        else:
            start = seg.get("start", 0.0)
            end = seg.get("end", 0.0)
            text = seg.get("text", "").strip()
            speaker = seg.get("speaker", "SPEAKER_00")
        if not text:
            continue
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
