"""CLI entry-point for TTRPG Story Narrator.

Usage::

    ttrpg-narrator narrate ./recordings --backend ollama --model llama3

Run ``ttrpg-narrator --help`` for full usage information.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Optional

import click

from ttrpg_narrator import __version__
from ttrpg_narrator import joiner, transcriber, writer


# ---------------------------------------------------------------------------
# CLI root
# ---------------------------------------------------------------------------

@click.group()
@click.version_option(__version__, prog_name="ttrpg-narrator")
def cli() -> None:
    """TTRPG Story Narrator — turn session recordings into narrative prose."""


# ---------------------------------------------------------------------------
# `narrate` command  (runs the full pipeline)
# ---------------------------------------------------------------------------

@cli.command()
@click.argument("input_folder", type=click.Path(exists=True, file_okay=False, path_type=Path))
@click.option(
    "--output", "-o",
    type=click.Path(dir_okay=False, writable=True, path_type=Path),
    default=None,
    help="Path for the output .md file.  Defaults to <input_folder>/story.md.",
)
@click.option(
    "--work-dir", "-w",
    type=click.Path(file_okay=False, path_type=Path),
    default=None,
    help="Working directory for intermediate files.  Defaults to <input_folder>/.narrator_work.",
)
@click.option(
    "--backend", "-b",
    type=click.Choice(["ollama", "gemini"], case_sensitive=False),
    default="ollama",
    show_default=True,
    help="LLM backend to use for narrative synthesis.",
)
@click.option(
    "--model", "-m",
    default=None,
    help=(
        "Model name for the chosen backend.  "
        "Defaults to 'llama3' for Ollama and 'gemini-1.5-pro' for Gemini."
    ),
)
@click.option(
    "--hf-token",
    envvar="HF_TOKEN",
    default=None,
    help=(
        "HuggingFace access token for pyannote diarization models "
        "(can also be set via HF_TOKEN).  "
        "If omitted, diarization is skipped."
    ),
)
@click.option(
    "--whisper-model",
    default="large-v2",
    show_default=True,
    help="WhisperX model size (e.g. 'large-v2', 'medium', 'small').",
)
@click.option(
    "--device",
    default="cpu",
    show_default=True,
    help="PyTorch device ('cpu', 'cuda', 'mps').",
)
@click.option(
    "--compute-type",
    default="int8",
    show_default=True,
    help="Quantization type ('int8', 'float16', 'float32').",
)
@click.option(
    "--num-speakers",
    type=int,
    default=None,
    help="Optional: exact number of speakers (improves diarization accuracy).",
)
@click.option(
    "--skip-join",
    is_flag=True,
    default=False,
    help="Skip Phase 1.  Expects 'full_session.wav' to already exist in --work-dir.",
)
@click.option(
    "--skip-transcribe",
    is_flag=True,
    default=False,
    help="Skip Phase 2.  Expects 'transcript.json' to already exist in --work-dir.",
)
@click.option(
    "--keep-work",
    is_flag=True,
    default=False,
    help="Keep intermediate files in --work-dir after completion.",
)
def narrate(
    input_folder: Path,
    output: Optional[Path],
    work_dir: Optional[Path],
    backend: str,
    model: Optional[str],
    hf_token: Optional[str],
    whisper_model: str,
    device: str,
    compute_type: str,
    num_speakers: Optional[int],
    skip_join: bool,
    skip_transcribe: bool,
    keep_work: bool,
) -> None:
    """Process TTRPG recordings in INPUT_FOLDER into a Markdown narrative.

    \b
    Phases:
      1. Join     — sort .m4a files by metadata, concatenate, convert to WAV
      2. Transcribe — WhisperX transcription + speaker diarization → JSON
      3. Write    — LLM cleans table talk, then writes narrative prose
      4. Output   — saves final story to a .md file
    """
    # ---- resolve defaults ----
    if work_dir is None:
        work_dir = input_folder / ".narrator_work"
    if output is None:
        output = input_folder / "story.md"
    if model is None:
        model = "llama3" if backend == "ollama" else "gemini-1.5-pro"

    # Read the Gemini API key from the environment only — passing it on the
    # command line would expose it in `ps aux` output and shell history.
    gemini_api_key: Optional[str] = os.environ.get("GEMINI_API_KEY") or None

    work_dir.mkdir(parents=True, exist_ok=True)

    wav_path = work_dir / "full_session.wav"
    transcript_path = work_dir / "transcript.json"

    # ---------------------------------------------------------------
    # Phase 1: Pre-processing
    # ---------------------------------------------------------------
    if skip_join:
        if not wav_path.exists():
            click.echo(
                f"[!] --skip-join was set but '{wav_path}' does not exist.", err=True
            )
            sys.exit(1)
        click.echo(f"[1/4] Skipping join — using existing '{wav_path}'.")
    else:
        click.echo("[1/4] Pre-processing audio files…")
        try:
            m4a_files = sorted(input_folder.glob("*.m4a"))
            click.echo(f"      Found {len(m4a_files)} .m4a file(s).")
            sorted_files = joiner.sort_files_chronologically(m4a_files)
            click.echo("      Joining and converting to WAV…")
            joiner.concatenate_audio(sorted_files, work_dir / "full_session.m4a")
            joiner.convert_to_wav(work_dir / "full_session.m4a", wav_path)
            click.echo(f"      WAV saved → {wav_path}")
        except FileNotFoundError as exc:
            click.echo(f"[!] {exc}", err=True)
            sys.exit(1)
        except Exception as exc:
            click.echo(f"[!] Phase 1 failed: {exc}", err=True)
            sys.exit(1)

    # ---------------------------------------------------------------
    # Phase 2: Transcription & Diarization
    # ---------------------------------------------------------------
    if skip_transcribe:
        if not transcript_path.exists():
            click.echo(
                f"[!] --skip-transcribe was set but '{transcript_path}' does not exist.",
                err=True,
            )
            sys.exit(1)
        click.echo(f"[2/4] Skipping transcription — using existing '{transcript_path}'.")
        segments = transcriber.load_transcript(transcript_path)
    else:
        click.echo("[2/4] Transcribing and diarizing…")
        if not hf_token:
            click.echo(
                "      [!] No HuggingFace token provided (--hf-token / HF_TOKEN). "
                "Speaker diarization will be skipped."
            )
        try:
            segments = transcriber.transcribe(
                wav_path,
                hf_token=hf_token,
                model_name=whisper_model,
                device=device,
                compute_type=compute_type,
                num_speakers=num_speakers,
            )
            transcriber.save_transcript(segments, transcript_path)
            click.echo(
                f"      Transcript saved → {transcript_path} ({len(segments)} segments)"
            )
        except Exception as exc:
            click.echo(f"[!] Phase 2 failed: {exc}", err=True)
            sys.exit(1)

    # ---------------------------------------------------------------
    # Phase 3 + 4: Narrative Synthesis & Output
    # ---------------------------------------------------------------
    click.echo(f"[3/4] Cleaning table talk with {backend}/{model}…")
    click.echo(f"[4/4] Generating narrative prose…")
    try:
        saved = writer.synthesize(
            segments,
            output_path=output,
            backend=backend,
            model=model,
            gemini_api_key=gemini_api_key,
        )
        click.echo(f"\n✓ Story saved → {saved}")
    except ValueError as exc:
        click.echo(f"[!] {exc}", err=True)
        sys.exit(1)
    except Exception as exc:
        click.echo(f"[!] Phase 3/4 failed: {exc}", err=True)
        sys.exit(1)

    # ---------------------------------------------------------------
    # Optional clean-up
    # ---------------------------------------------------------------
    if not keep_work:
        import shutil
        if work_dir.is_symlink():
            click.echo(
                f"[!] Skipping cleanup: '{work_dir}' is a symlink. "
                "Remove it manually if desired.",
                err=True,
            )
        else:
            shutil.rmtree(work_dir)


# ---------------------------------------------------------------------------
# `join` sub-command  (Phase 1 only)
# ---------------------------------------------------------------------------

@cli.command()
@click.argument("input_folder", type=click.Path(exists=True, file_okay=False, path_type=Path))
@click.option(
    "--work-dir", "-w",
    type=click.Path(file_okay=False, path_type=Path),
    default=None,
    help="Directory for output files.  Defaults to <input_folder>/.narrator_work.",
)
def join(input_folder: Path, work_dir: Optional[Path]) -> None:
    """Phase 1 only: join and convert .m4a files to WAV."""
    if work_dir is None:
        work_dir = input_folder / ".narrator_work"

    click.echo(f"Scanning '{input_folder}' for .m4a files…")
    try:
        wav_path = joiner.process_folder(input_folder, work_dir)
        click.echo(f"✓ WAV saved → {wav_path}")
    except FileNotFoundError as exc:
        click.echo(f"[!] {exc}", err=True)
        sys.exit(1)
    except Exception as exc:
        click.echo(f"[!] Join failed: {exc}", err=True)
        sys.exit(1)


# ---------------------------------------------------------------------------
# `transcribe` sub-command  (Phase 2 only)
# ---------------------------------------------------------------------------

@cli.command()
@click.argument("wav_file", type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.option(
    "--output", "-o",
    type=click.Path(dir_okay=False, writable=True, path_type=Path),
    default=None,
    help="Output JSON path.  Defaults to <wav_file stem>.json in the same directory.",
)
@click.option("--hf-token", envvar="HF_TOKEN", default=None)
@click.option("--whisper-model", default="large-v2", show_default=True)
@click.option("--device", default="cpu", show_default=True)
@click.option("--compute-type", default="int8", show_default=True)
@click.option("--num-speakers", type=int, default=None)
def transcribe(
    wav_file: Path,
    output: Optional[Path],
    hf_token: Optional[str],
    whisper_model: str,
    device: str,
    compute_type: str,
    num_speakers: Optional[int],
) -> None:
    """Phase 2 only: transcribe and diarize a WAV file → JSON."""
    if output is None:
        output = wav_file.with_suffix(".json")

    if not hf_token:
        click.echo(
            "[!] No HuggingFace token provided. Speaker diarization will be skipped."
        )

    click.echo(f"Transcribing '{wav_file}'…")
    try:
        segments = transcriber.transcribe(
            wav_file,
            hf_token=hf_token,
            model_name=whisper_model,
            device=device,
            compute_type=compute_type,
            num_speakers=num_speakers,
        )
        transcriber.save_transcript(segments, output)
        click.echo(f"✓ Transcript saved → {output} ({len(segments)} segments)")
    except Exception as exc:
        click.echo(f"[!] Transcription failed: {exc}", err=True)
        sys.exit(1)


# ---------------------------------------------------------------------------
# `write` sub-command  (Phase 3+4 only)
# ---------------------------------------------------------------------------

@cli.command()
@click.argument("transcript_json", type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.option(
    "--output", "-o",
    type=click.Path(dir_okay=False, writable=True, path_type=Path),
    default=None,
    help="Output .md path.  Defaults to <json stem>.md in the same directory.",
)
@click.option(
    "--backend", "-b",
    type=click.Choice(["ollama", "gemini"], case_sensitive=False),
    default="ollama",
    show_default=True,
)
@click.option("--model", "-m", default=None)
def write(
    transcript_json: Path,
    output: Optional[Path],
    backend: str,
    model: Optional[str],
) -> None:
    """Phase 3+4 only: synthesize narrative prose from a transcript JSON."""
    if output is None:
        output = transcript_json.with_suffix(".md")
    if model is None:
        model = "llama3" if backend == "ollama" else "gemini-1.5-pro"

    # Read the Gemini API key from the environment only.
    gemini_api_key: Optional[str] = os.environ.get("GEMINI_API_KEY") or None

    click.echo(f"Loading transcript from '{transcript_json}'…")
    try:
        segments = transcriber.load_transcript(transcript_json)
    except Exception as exc:
        click.echo(f"[!] Failed to load transcript: {exc}", err=True)
        sys.exit(1)

    click.echo(f"Cleaning table talk with {backend}/{model}…")
    click.echo("Generating narrative…")
    try:
        saved = writer.synthesize(
            segments,
            output_path=output,
            backend=backend,
            model=model,
            gemini_api_key=gemini_api_key,
        )
        click.echo(f"✓ Story saved → {saved}")
    except ValueError as exc:
        click.echo(f"[!] {exc}", err=True)
        sys.exit(1)
    except Exception as exc:
        click.echo(f"[!] Write failed: {exc}", err=True)
        sys.exit(1)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    cli()
