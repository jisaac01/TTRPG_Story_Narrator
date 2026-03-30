"""Phase 3 — Narrative Synthesis (The "Writer") + Phase 4 — Output.

Two-step LLM pipeline:

* **Step A — The Cleaner**: strips out-of-character "table talk" from the
  diarized transcript.
* **Step B — The Storyteller**: converts the cleaned transcript into
  third-person narrative prose.

Supports two backends selected via the ``backend`` parameter:

* ``"ollama"``  — local Ollama server (no API key needed).
* ``"gemini"``  — Google Gemini API (requires ``GEMINI_API_KEY``).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import List, Optional

from ttrpg_narrator.transcriber import Segment


# ---------------------------------------------------------------------------
# Prompt templates
# ---------------------------------------------------------------------------

_CLEANER_PROMPT = """\
You are an editor processing a transcript from a tabletop RPG (TTRPG) session.
Your task is to remove every "out-of-character" or "table talk" exchange — \
things like passing snacks, rules questions ("What's the AC?"), \
side conversations, or anything unrelated to the story being played.

Below is the transcript as a JSON array.  Each element has the keys:
  "start", "end", "speaker", "text"

Return ONLY the segments that contain in-character actions, dialogue, or \
story-relevant content.  Output valid JSON using the exact same schema — \
no extra keys, no commentary, no markdown fences.

Transcript:
{transcript}
"""

_STORYTELLER_PROMPT = """\
You are a skilled fantasy author.  Below is a filtered transcript from a \
tabletop RPG session, presented as a JSON array with speaker labels \
(e.g. SPEAKER_00, SPEAKER_01).

Your task: convert this transcript into richly written, third-person \
narrative prose.  Use the context to infer character names where possible; \
otherwise refer to them as "the first adventurer", "the second adventurer", \
etc.  Merge sequential actions and dialogue into flowing paragraphs.  \
Do NOT simply list events — write it as an engaging story.

Transcript:
{transcript}

Write the narrative story below:
"""


# ---------------------------------------------------------------------------
# Backend helpers
# ---------------------------------------------------------------------------

def _call_ollama(prompt: str, model: str) -> str:
    """Send *prompt* to a local Ollama instance and return the response text."""
    try:
        import ollama  # type: ignore
    except ImportError as exc:
        raise ImportError(
            "ollama is not installed. Run: pip install ollama"
        ) from exc

    response = ollama.chat(
        model=model,
        messages=[{"role": "user", "content": prompt}],
    )
    return response["message"]["content"]


def _call_gemini(prompt: str, model: str, api_key: str) -> str:
    """Send *prompt* to the Gemini API and return the response text."""
    try:
        import google.generativeai as genai  # type: ignore
    except ImportError as exc:
        raise ImportError(
            "google-generativeai is not installed. Run: pip install google-generativeai"
        ) from exc

    genai.configure(api_key=api_key)
    gemini_model = genai.GenerativeModel(model)
    response = gemini_model.generate_content(prompt)
    return response.text


def _llm_call(
    prompt: str,
    backend: str,
    model: str,
    gemini_api_key: Optional[str] = None,
) -> str:
    """Dispatch an LLM call to the chosen *backend*."""
    if backend == "ollama":
        return _call_ollama(prompt, model)
    elif backend == "gemini":
        if not gemini_api_key:
            raise ValueError(
                "A Gemini API key is required when using the 'gemini' backend. "
                "Pass --gemini-api-key or set the GEMINI_API_KEY environment variable."
            )
        return _call_gemini(prompt, model, gemini_api_key)
    else:
        raise ValueError(f"Unknown backend '{backend}'. Choose 'ollama' or 'gemini'.")


# ---------------------------------------------------------------------------
# Pipeline steps
# ---------------------------------------------------------------------------

def clean_table_talk(
    segments: List[Segment],
    backend: str,
    model: str,
    gemini_api_key: Optional[str] = None,
) -> List[Segment]:
    """Step A: Remove out-of-character segments using an LLM.

    Returns a filtered list of segments.  Falls back to returning the
    original list unchanged if the LLM response cannot be parsed.
    """
    transcript_json = json.dumps(segments, indent=2, ensure_ascii=False)
    prompt = _CLEANER_PROMPT.format(transcript=transcript_json)
    raw = _llm_call(prompt, backend, model, gemini_api_key)

    # Strip potential markdown fences from the LLM response
    text = raw.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        # Remove the opening fence line (e.g. ```json or ```)
        if lines and lines[0].strip().startswith("```"):
            lines = lines[1:]
        # Remove the closing fence line if present
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        text = "\n".join(lines)

    try:
        cleaned: List[Segment] = json.loads(text)
        if not isinstance(cleaned, list):
            raise ValueError("Expected a JSON array.")
        return cleaned
    except (json.JSONDecodeError, ValueError):
        # Gracefully degrade: return original segments
        return segments


def generate_narrative(
    segments: List[Segment],
    backend: str,
    model: str,
    gemini_api_key: Optional[str] = None,
) -> str:
    """Step B: Turn cleaned segments into third-person narrative prose."""
    transcript_json = json.dumps(segments, indent=2, ensure_ascii=False)
    prompt = _STORYTELLER_PROMPT.format(transcript=transcript_json)
    return _llm_call(prompt, backend, model, gemini_api_key)


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------

def save_narrative(narrative: str, output_path: Path) -> Path:
    """Write *narrative* to a Markdown file at *output_path*."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as fh:
        fh.write(narrative)
        if not narrative.endswith("\n"):
            fh.write("\n")
    return output_path


# ---------------------------------------------------------------------------
# Convenience wrapper
# ---------------------------------------------------------------------------

def synthesize(
    segments: List[Segment],
    output_path: Path,
    backend: str = "ollama",
    model: str = "llama3",
    gemini_api_key: Optional[str] = None,
) -> Path:
    """Run the full Phase 3+4 pipeline and save a Markdown narrative.

    Parameters
    ----------
    segments:
        Diarized transcript segments from Phase 2.
    output_path:
        Destination ``.md`` file.
    backend:
        ``"ollama"`` or ``"gemini"``.
    model:
        Model name passed to the chosen backend.
    gemini_api_key:
        Required when *backend* is ``"gemini"``.

    Returns
    -------
    Path
        The path to the saved Markdown file.
    """
    cleaned = clean_table_talk(segments, backend, model, gemini_api_key)
    narrative = generate_narrative(cleaned, backend, model, gemini_api_key)
    return save_narrative(narrative, output_path)
