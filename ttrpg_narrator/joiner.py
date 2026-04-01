"""Phase 1 — Pre-processing (The "Joiner").

Finds all .m4a files in a folder, sorts them chronologically using
"Media Created" metadata, concatenates them with ffmpeg, and converts
the result to a 16 kHz mono WAV suitable for transcription.
"""

from __future__ import annotations

import json
import subprocess
import tempfile
from pathlib import Path
from typing import List, Optional


# ---------------------------------------------------------------------------
# Metadata helpers
# ---------------------------------------------------------------------------

def _get_creation_time_ffprobe(file_path: Path) -> Optional[str]:
    """Return the creation_time tag reported by ffprobe, or None."""
    try:
        result = subprocess.run(
            [
                "ffprobe", "-v", "quiet",
                "-print_format", "json",
                "-show_format",
                str(file_path),
            ],
            capture_output=True,
            text=True,
            check=True,
        )
        data = json.loads(result.stdout)
        tags = data.get("format", {}).get("tags", {})
        for key in ("creation_time", "com.apple.quicktime.creationdate", "©day"):
            if key in tags:
                return tags[key]
    except (subprocess.CalledProcessError, json.JSONDecodeError, FileNotFoundError):
        pass
    return None


def get_media_created(file_path: Path) -> Optional[str]:
    """Return an ISO-8601-ish creation timestamp for *file_path*, or None.

    Tries mutagen first (faster, no subprocess), then falls back to ffprobe.
    """
    # mutagen path
    try:
        from mutagen.mp4 import MP4  # type: ignore

        audio = MP4(str(file_path))
        if audio.tags:
            for tag in ("©day", "com.apple.quicktime.creationdate"):
                values = audio.tags.get(tag)
                if values:
                    return str(values[0])
    except Exception:
        pass

    # ffprobe fallback
    return _get_creation_time_ffprobe(file_path)


# ---------------------------------------------------------------------------
# Sorting
# ---------------------------------------------------------------------------

def sort_files_chronologically(files: List[Path]) -> List[Path]:
    """Sort *.m4a* files by Media Created metadata, falling back to filename.

    Files with a detected timestamp sort before files without one.
    """

    def sort_key(f: Path):
        ts = get_media_created(f)
        if ts:
            return (0, ts, f.name)
        return (1, "", f.name)

    return sorted(files, key=sort_key)


# ---------------------------------------------------------------------------
# ffmpeg helpers
# ---------------------------------------------------------------------------

def concatenate_audio(files: List[Path], output_path: Path) -> Path:
    """Concatenate *files* into *output_path* (.m4a) using ffmpeg concat demuxer."""
    if len(files) == 1:
        subprocess.run(
            ["ffmpeg", "-y", "-i", str(files[0]), "-c", "copy",
             "-movflags", "+faststart", str(output_path)],
            check=True,
            capture_output=True,
        )
        return output_path

    # Build a temporary concat list
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".txt", delete=False
    ) as list_file:
        list_path = Path(list_file.name)
        for fp in files:
            # Escape backslashes then single-quotes so a filename like
            # a'.m4a cannot break out of the quoted path and inject extra
            # directives into the ffmpeg concat list.
            escaped = str(fp.resolve()).replace("\\", "\\\\").replace("'", "\\'")
            list_file.write(f"file '{escaped}'\n")

    try:
        subprocess.run(
            [
                "ffmpeg", "-y",
                "-f", "concat", "-safe", "0",
                "-i", str(list_path),
                "-c", "copy",
                "-movflags", "+faststart",
                str(output_path),
            ],
            check=True,
            capture_output=True,
        )
    finally:
        list_path.unlink(missing_ok=True)

    return output_path


def convert_to_wav(input_path: Path, output_path: Path, sample_rate: int = 16000) -> Path:
    """Convert *input_path* to a mono WAV at *sample_rate* Hz."""
    subprocess.run(
        [
            "ffmpeg", "-y",
            "-i", str(input_path),
            "-ac", "1",
            "-ar", str(sample_rate),
            "-vn",
            str(output_path),
        ],
        check=True,
        capture_output=True,
    )
    return output_path


# ---------------------------------------------------------------------------
# Public entry-point
# ---------------------------------------------------------------------------

def process_folder(input_folder: Path, work_dir: Path) -> Path:
    """Run the full Phase 1 pipeline.

    1. Discover all *.m4a* files in *input_folder*.
    2. Sort them chronologically.
    3. Concatenate into ``work_dir/full_session.m4a``.
    4. Convert to ``work_dir/full_session.wav`` (16 kHz mono).

    Returns the path to the WAV file.
    """
    m4a_files = sorted(input_folder.glob("*.m4a"))
    if not m4a_files:
        raise FileNotFoundError(f"No .m4a files found in '{input_folder}'.")

    sorted_files = sort_files_chronologically(m4a_files)

    work_dir.mkdir(parents=True, exist_ok=True)

    full_session_m4a = work_dir / "full_session.m4a"
    concatenate_audio(sorted_files, full_session_m4a)

    wav_path = work_dir / "full_session.wav"
    convert_to_wav(full_session_m4a, wav_path)

    return wav_path
