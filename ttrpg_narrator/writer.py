"""Phase 3 — Narrative Synthesis (The "Writer") + Phase 4 — Output.

Four-pass LLM pipeline:

* **Pass 1 — Speaker ID**: identifies which SPEAKER_XX is the GM and maps
  players to character names from a short head-of-transcript sample.
* **Pass 2 — Catalogue extraction**: chunks the transcript and extracts
  structured game info (PCs, NPCs, locations, items, enemies, quotes, etc.).
* **Pass 3 — Scene outline**: chunks the transcript and writes a
  bullet-point scene summary per chunk, then concatenates them.
* **Pass 4 — Final recap**: uses the outline + catalogue (small context,
  fits any model) to write the final session summary.

Also includes the legacy ``clean_table_talk`` step (Pass 0) which strips
out-of-character table talk from the raw transcript segments.

Supports two backends selected via the ``backend`` parameter:

* ``"ollama"``  — local Ollama server (no API key needed).
* ``"gemini"``  — Google Gemini API (requires ``GEMINI_API_KEY``).
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import List, Optional

from ttrpg_narrator.transcriber import Segment


# ---------------------------------------------------------------------------
# Prompt templates (system role only — transcript is passed as user content)
# ---------------------------------------------------------------------------

# Keeping instructions in the system role and transcript data in the user
# role prevents prompt-injection attacks where adversarial content in the
# audio transcript could override these instructions.

_CLEANER_SYSTEM = """\
You are an editor processing a transcript from a tabletop RPG (TTRPG) session.
Your task is to remove clearly "out-of-character" or "table talk" exchanges — \
things like passing snacks, pizza orders, bathroom breaks, rules arguments \
("What's the AC?"), phone interruptions, or pure meta-game chatter that has \
NOTHING to do with the story being played.

The user will supply the transcript as a JSON array.  Each element has the keys:
  "start", "end", "speaker", "text"

Be CONSERVATIVE: when in doubt, keep the segment.  It is far better to \
include a borderline segment than to remove story-relevant content. \
Only remove segments that are unambiguously out-of-character.  \
The output must contain at least as many segments as were clearly in-character; \
never return an empty array.

Return ONLY the retained segments using the exact same JSON schema — \
no extra keys, no commentary, no markdown fences.
"""

_SPEAKER_ID_SYSTEM = """\
You are analysing the opening of a tabletop RPG (TTRPG) session transcript.
Your task is to identify each unique speaker ID and classify them.
The transcript is provided as plain text lines: "SPEAKER_XX: <what they said>".
One speaker is the Game Master (GM) — they describe the world, narrate outcomes,
and voice NPCs.  The others are players speaking as or about their characters.

Return a JSON object where each key is a speaker ID (e.g. "SPEAKER_05") and
the value is an object with:
  "role": "GM" | "player"
  "character": "<character name if detectable, else null>"  (players only)

Return ONLY the JSON object — no commentary, no markdown fences.
"""

_CATALOGUE_SYSTEM = """\
You are extracting structured information from a chunk of a tabletop RPG transcript.
{speaker_context}

The transcript chunk is provided as plain text lines: "SPEAKER_XX: <what they said>".
Extract and return a JSON object — use empty lists/strings where nothing is mentioned:
  "pcs"           : list of {{"name": str, "speaker": str, "notes": str}}
  "npcs"          : list of {{"name": str, "role": str, "notes": str}}
  "setting"       : str  (world/setting description, or "")
  "locations"     : list of {{"name": str, "scale": "macro|micro", "description": str}}
  "items"         : list of {{"name": str, "found_by": str, "notes": str}}
  "enemies"       : list of {{"name": str, "encounter": str}}
  "puzzles"       : list of {{"description": str, "resolution": str}}
  "quotes"        : list of {{"speaker": str, "character": str, "text": str}}
  "quest_goal"    : str  (what the party is trying to achieve, or "")
  "key_decisions" : list of {{"description": str, "outcome": str}}

Return ONLY the JSON object — no commentary, no markdown fences.
"""

_OUTLINE_SYSTEM = """\
You are summarising one chunk of a tabletop RPG session transcript into a scene recap.
{speaker_context}

Known session catalogue (for context — do not repeat it verbatim):
{catalogue_context}

The transcript chunk is provided as plain text lines: "SPEAKER_XX: <what they said>".
The speaker marked as GM (or the one describing the world/NPCs) is the narrator.
Treat their lines as objective scene description — what the characters see, hear, and
experience — not as things the GM "said".

Write 3–8 bullet points describing what happened in this chunk.
Rules:
- Use GM narration as factual scene description.
- Use character names from the catalogue where known; otherwise use "the party" or
  "the adventurers".
- Dice rolls and mechanical outcomes must be converted to story results.
  Example: instead of "rolled a 14 vs AC 8", write "landed a solid hit" or
  "barely dodged the blow". Convey the narrative outcome, not the number.
- Include key decisions, dialogue, discoveries, and consequences.
- Each bullet is one concise sentence.
- Begin your response IMMEDIATELY with the first bullet (- or •).
  Do NOT write any intro sentence, heading, or preamble whatsoever.
"""

_COMPRESS_SYSTEM = """\
You are condensing a TTRPG session scene outline into a shorter summary.

You will receive a set of bullet points from one portion of the session.
Condense them into 3–5 bullets that preserve:
- Character names and what they did
- Key outcomes (victories, losses, discoveries)
- Story-significant moments

Remove redundant bullets, dice roll numbers, and meta-game references.
Your response MUST start immediately with the first bullet (- or •). No preamble.
"""

_CONTINUITY_SYSTEM = """\
You are a continuity editor reviewing a TTRPG session recap against its source outline.

You will receive:
  OUTLINE: bullet points of what actually happened in the session.
  DRAFT: the recap written from that outline.

Check the draft against the outline for:
- Events described incorrectly (e.g. draft says "unharmed" when outline says "killed")
- Key events that are missing from the draft
- Wrong character names or outcomes

Correct any errors and return the full rewritten recap.
Keep the same format and structure as the draft.
Do not add invented details — only fix factual contradictions with the outline.
Return ONLY the corrected recap, no commentary.
"""

_STORYTELLER_SYSTEM = """\
You are writing a session recap for a tabletop RPG group to read after their game.

You will receive a scene-by-scene OUTLINE and a structured CATALOGUE of
characters, locations, items, and other key details from the session.

Never mention the Game Master, players, dice rolls, game mechanics, or anything
outside the story world. Write entirely from within the fiction.

Write the recap in this exact format:

1. A session title on the first line (e.g. "Session 6 — The Sunken Vault").
2. Two to four narrative paragraphs summarising events chronologically.
3. Then the following labelled sections, each heading on its own line in ALL CAPS:

OUTLINE:
- One bullet per scene, summarising what happened.

NPCs:
- Name: brief in-world description.

ITEMS:
- Name: brief description.

LOCATIONS:
- Name: brief description.

QUESTS:
- Quest name: what the characters are trying to accomplish.

PLAYER CHARACTERS
- Name: role, notable abilities or traits shown in this session.

Use only information present in the outline and catalogue. Do not invent details.
"""


# ---------------------------------------------------------------------------
# Backend helpers
# ---------------------------------------------------------------------------

def _call_ollama(system: str, user: str, model: str) -> str:
    """Send a system+user message pair to a local Ollama instance."""
    try:
        import ollama  # type: ignore
    except ImportError as exc:
        raise ImportError(
            "ollama is not installed. Run: pip install ollama"
        ) from exc

    response = ollama.chat(
        model=model,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
    )
    return response["message"]["content"]


def _call_gemini(system: str, user: str, model: str, api_key: str) -> str:
    """Send a system+user message pair to the Gemini API."""
    try:
        import google.generativeai as genai  # type: ignore
    except ImportError as exc:
        raise ImportError(
            "google-generativeai is not installed. Run: pip install google-generativeai"
        ) from exc

    genai.configure(api_key=api_key)
    gemini_model = genai.GenerativeModel(model, system_instruction=system)
    response = gemini_model.generate_content(user)
    return response.text


def _llm_call(
    system: str,
    user: str,
    backend: str,
    model: str,
    gemini_api_key: Optional[str] = None,
) -> str:
    """Dispatch an LLM call to the chosen *backend*."""
    if backend == "ollama":
        return _call_ollama(system, user, model)
    elif backend == "gemini":
        if not gemini_api_key:
            raise ValueError(
                "A Gemini API key is required when using the 'gemini' backend. "
                "Set the GEMINI_API_KEY environment variable."
            )
        return _call_gemini(system, user, model, gemini_api_key)
    else:
        raise ValueError(f"Unknown backend '{backend}'. Choose 'ollama' or 'gemini'.")


def _strip_fences(text: str) -> str:
    """Remove markdown code fences (```...```) from an LLM response."""
    text = text.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        if lines and lines[0].strip().startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        text = "\n".join(lines)
    return text


def _extract_json_object(text: str) -> dict:
    """Extract and parse the first JSON object ``{...}`` from an LLM response.

    Handles the common llama3 pattern where the model wraps the JSON dict in
    prose commentary both before and after.  Falls back to ``{}`` on failure.
    """
    text = _strip_fences(text).strip()
    # Fast path: the whole text is valid JSON
    try:
        result = json.loads(text)
        if isinstance(result, dict):
            return result
    except (json.JSONDecodeError, ValueError):
        pass
    # Slow path: find the outermost {...} block
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        try:
            result = json.loads(match.group(0))
            if isinstance(result, dict):
                return result
        except (json.JSONDecodeError, ValueError):
            pass
    return {}


# Regex that matches common LLM preamble lines the models add before bullet lists.
_OUTLINE_PREAMBLE_RE = re.compile(
    r"^(here (are|is|'?s)|this (chunk|scene|section|part)( covers)?|these are"
    r"|the following|below (are|is))",
    re.IGNORECASE,
)


def _strip_outline_preamble(text: str) -> str:
    """Remove leading preamble lines from LLM outline chunk output.

    Strips lines matching common introductory patterns such as
    "Here are the bullet points summarizing what happened in this chunk:".
    """
    lines = text.strip().splitlines()
    while lines and _OUTLINE_PREAMBLE_RE.match(lines[0].strip()):
        lines.pop(0)
    # Drop any blank lines after the stripped preamble
    while lines and not lines[0].strip():
        lines.pop(0)
    return "\n".join(lines)


def _format_speaker_context(speaker_map: dict) -> str:
    """Format the speaker map as a readable context string for prompts."""
    if not speaker_map:
        return "Speaker roles are unknown."
    lines = ["Known speakers:"]
    for sid, info in speaker_map.items():
        role = info.get("role", "unknown")
        char = info.get("character")
        if char:
            lines.append(f"  {sid}: {role} (character: {char})")
        else:
            lines.append(f"  {sid}: {role}")
    return "\n".join(lines)


def _merge_catalogues(catalogues: List[dict]) -> dict:
    """Merge per-chunk catalogue dicts into a single combined catalogue.

    List fields are concatenated; string fields keep the first non-empty value.
    """
    merged: dict = {
        "pcs": [], "npcs": [], "locations": [], "items": [],
        "enemies": [], "puzzles": [], "quotes": [], "key_decisions": [],
        "setting": "", "quest_goal": "",
    }
    for cat in catalogues:
        for key in ("pcs", "npcs", "locations", "items", "enemies",
                    "puzzles", "quotes", "key_decisions"):
            merged[key].extend(cat.get(key, []))
        for key in ("setting", "quest_goal"):
            if not merged[key] and cat.get(key):
                merged[key] = cat[key]
    return merged


# ---------------------------------------------------------------------------
# Pipeline steps
# ---------------------------------------------------------------------------

def clean_table_talk(
    segments: List[Segment],
    backend: str,
    model: str,
    gemini_api_key: Optional[str] = None,
) -> List[Segment]:
    """Pass 0: Remove out-of-character segments using an LLM.

    Returns a filtered list of segments.  Falls back to returning the
    original list unchanged if the LLM response cannot be parsed.
    """
    transcript_json = json.dumps(segments, indent=2, ensure_ascii=False)
    raw = _llm_call(_CLEANER_SYSTEM, transcript_json, backend, model, gemini_api_key)
    text = _strip_fences(raw)

    # Guard against excessively large LLM responses before parsing.
    if len(text) > 10 * len(transcript_json):
        return segments

    try:
        cleaned: List[Segment] = json.loads(text)
        if not isinstance(cleaned, list):
            raise ValueError("Expected a JSON array.")
        if not cleaned:
            return segments
        return cleaned
    except (json.JSONDecodeError, ValueError):
        return segments


def identify_speakers(
    segments: List[Segment],
    backend: str,
    model: str,
    gemini_api_key: Optional[str] = None,
    head_size: int = 150,
) -> dict:
    """Pass 1: Identify which SPEAKER_XX is the GM and map players to character names.

    Only the first *head_size* segments are sent.  The opening of the session
    is usually enough to distinguish the GM's descriptive voice from the
    players'.  Returns an empty dict on parse failure.
    """
    # Build the minimum head that covers every unique speaker in the transcript,
    # but always include at least head_size segments for context.
    all_speakers = {s["speaker"] for s in segments}
    seen: set = set()
    head: List[Segment] = []
    for seg in segments:
        head.append(seg)
        seen.add(seg["speaker"])
        if seen >= all_speakers and len(head) >= head_size:
            break
    # If coverage was reached before head_size, pad to head_size for context.
    if len(head) < head_size:
        head = segments[:head_size]
    plain_text = "\n".join(f"{s['speaker']}: {s['text']}" for s in head)
    raw = _llm_call(_SPEAKER_ID_SYSTEM, plain_text, backend, model, gemini_api_key)
    return _extract_json_object(raw)


def extract_catalogue(
    segments: List[Segment],
    speaker_map: dict,
    backend: str,
    model: str,
    gemini_api_key: Optional[str] = None,
    chunk_size: int = 100,
) -> dict:
    """Pass 2: Extract structured game info from the transcript in chunks.

    The transcript is split into chunks of *chunk_size* segments.  Each chunk
    is processed independently and the results merged.  Chunking keeps every
    call within the model's context window even for long sessions.

    Returns a catalogue dict with keys: pcs, npcs, setting, locations, items,
    enemies, puzzles, quotes, quest_goal, key_decisions.
    """
    empty: dict = {
        "pcs": [], "npcs": [], "locations": [], "items": [],
        "enemies": [], "puzzles": [], "quotes": [], "key_decisions": [],
        "setting": "", "quest_goal": "",
    }
    speaker_context = _format_speaker_context(speaker_map)
    system = _CATALOGUE_SYSTEM.format(speaker_context=speaker_context)

    chunks = [segments[i:i + chunk_size] for i in range(0, len(segments), chunk_size)]
    partial: List[dict] = []
    for chunk in chunks:
        plain_text = "\n".join(f"{s['speaker']}: {s['text']}" for s in chunk)
        raw = _llm_call(system, plain_text, backend, model, gemini_api_key)
        text = _strip_fences(raw)
        try:
            cat = json.loads(text)
            if isinstance(cat, dict):
                partial.append(cat)
        except (json.JSONDecodeError, ValueError):
            pass

    if not partial:
        return empty
    return _merge_catalogues(partial)


def generate_outline(
    segments: List[Segment],
    speaker_map: dict,
    catalogue: dict,
    backend: str,
    model: str,
    gemini_api_key: Optional[str] = None,
    chunk_size: int = 100,
) -> str:
    """Pass 3: Build a scene-by-scene outline by summarising transcript chunks.

    Each chunk produces a short bullet-point scene summary.  The summaries
    are joined in order to form a chronological outline of the full session.
    """
    speaker_context = _format_speaker_context(speaker_map)
    catalogue_context = json.dumps(catalogue, indent=2, ensure_ascii=False)
    system = _OUTLINE_SYSTEM.format(
        speaker_context=speaker_context,
        catalogue_context=catalogue_context,
    )

    chunks = [segments[i:i + chunk_size] for i in range(0, len(segments), chunk_size)]
    scene_summaries: List[str] = []
    for chunk in chunks:
        plain_text = "\n".join(f"{s['speaker']}: {s['text']}" for s in chunk)
        raw = _llm_call(system, plain_text, backend, model, gemini_api_key)
        scene_summaries.append(_strip_outline_preamble(raw))

    return "\n\n".join(scene_summaries)


def generate_narrative(
    catalogue: dict,
    outline: str,
    backend: str,
    model: str,
    gemini_api_key: Optional[str] = None,
) -> str:
    """Pass 4: Write the final session recap from the outline and catalogue.

    Using only the pre-built outline and catalogue (not the raw transcript)
    keeps the input small — it fits any model's context window — and prevents
    the summary from being silently truncated mid-session.
    """
    user_content = (
        f"## Session Outline\n{outline}\n\n"
        f"## Catalogue\n{json.dumps(catalogue, indent=2, ensure_ascii=False)}"
    )
    return _llm_call(_STORYTELLER_SYSTEM, user_content, backend, model, gemini_api_key)


def normalize_transcript(segments: List[dict]) -> List[dict]:
    """Pre-process: merge consecutive same-speaker segments into screenplay lines.

    Removes timestamps and merges runs of the same speaker so the transcript
    resembles a screenplay rather than a list of short JSON objects.  This
    dramatically reduces token count (a 3,000-segment session typically
    compresses to ~500 merged lines) and gives the LLM more context per call.

    Empty or whitespace-only segments are silently dropped.
    Timestamps are stripped from the output (``start``/``end`` keys removed).
    """
    if not segments:
        return []

    merged: List[dict] = []
    current_speaker: str = segments[0]["speaker"]
    buffer: List[str] = []

    for seg in segments:
        text = seg["text"].strip()
        if not text:
            continue
        if seg["speaker"] == current_speaker:
            buffer.append(text)
        else:
            if buffer:
                merged.append({"speaker": current_speaker, "text": " ".join(buffer)})
            current_speaker = seg["speaker"]
            buffer = [text]

    if buffer:
        merged.append({"speaker": current_speaker, "text": " ".join(buffer)})

    return merged


def compress_outline(
    outline: str,
    backend: str,
    model: str,
    gemini_api_key: Optional[str] = None,
    chunk_size: int = 20,
) -> str:
    """Pass 3.5: Compress the full session outline to fit the narrative context window.

    The outline produced by ``generate_outline`` can be 5,000+ tokens — larger
    than llama3's context.  This pass chunks the outline into groups of
    *chunk_size* lines, compresses each group to 3–5 key events, and joins
    the results.  The compressed outline is typically under 500 tokens.
    """
    lines = [ln for ln in outline.splitlines() if ln.strip()]
    if not lines:
        return outline

    chunks = [lines[i:i + chunk_size] for i in range(0, len(lines), chunk_size)]
    compressed_parts: List[str] = []
    for chunk in chunks:
        chunk_text = "\n".join(chunk)
        result = _llm_call(_COMPRESS_SYSTEM, chunk_text, backend, model, gemini_api_key)
        compressed_parts.append(_strip_outline_preamble(result).strip())

    return "\n".join(compressed_parts)


def continuity_check(
    narrative: str,
    outline: str,
    backend: str,
    model: str,
    gemini_api_key: Optional[str] = None,
) -> str:
    """Pass 4.5: Review the narrative against the outline and fix contradictions.

    Sends the compressed outline and the generated narrative to the LLM and
    asks it to identify any factual errors (e.g. "children were unharmed" when
    the outline says they were killed) and rewrite the narrative with corrections.
    """
    user_content = (
        f"OUTLINE:\n{outline}\n\n"
        f"DRAFT:\n{narrative}"
    )
    return _llm_call(_CONTINUITY_SYSTEM, user_content, backend, model, gemini_api_key)


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
    clean_cache_path: Optional[Path] = None,
    speaker_cache_path: Optional[Path] = None,
    catalogue_cache_path: Optional[Path] = None,
    outline_cache_path: Optional[Path] = None,
    compressed_outline_cache_path: Optional[Path] = None,
    chunk_size: int = 100,
    log=None,
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
    clean_cache_path:
        Optional path to cache the table-talk-cleaned segments JSON.
    speaker_cache_path:
        Optional path to cache the speaker identification JSON.
    catalogue_cache_path:
        Optional path to cache the extracted catalogue JSON.
    outline_cache_path:
        Optional path to cache the generated scene outline.
    compressed_outline_cache_path:
        Optional path to cache the compressed (context-window-safe) outline.
    chunk_size:
        Number of transcript segments per LLM call for chunked passes
        (catalogue extraction and outline generation).  Keep at ≤100 for
        models with a 4K context window; raise to 500+ for 128K+ models.
    log:
        Callable used for progress messages (default: print).

    Returns
    -------
    Path
        The path to the saved Markdown file.
    """
    if log is None:
        log = print

    # --- Pre-processing: normalize transcript ---
    log(f"      [Pre] Normalizing transcript ({len(segments)} raw segments)…")
    normalized = normalize_transcript(segments)
    log(f"      [Pre] Normalized → {len(normalized)} merged lines.")

    # --- Pass 0: Clean table talk (operates on normalized segments) ---
    if clean_cache_path is not None and clean_cache_path.exists():
        log(f"      [Pass 0] Loading cached cleaned transcript from {clean_cache_path}…")
        with open(clean_cache_path, encoding="utf-8") as _fh:
            cleaned = json.load(_fh)
        log(f"      [Pass 0] {len(cleaned)} segments loaded (skipping LLM clean).")
    else:
        log(f"      [Pass 0] Cleaning table talk with {backend}/{model}…")
        cleaned = clean_table_talk(normalized, backend, model, gemini_api_key)
        log(f"      [Pass 0] Done — {len(cleaned)} segments remain.")
        if clean_cache_path is not None:
            clean_cache_path.parent.mkdir(parents=True, exist_ok=True)
            with open(clean_cache_path, "w", encoding="utf-8") as _fh:
                json.dump(cleaned, _fh, indent=2, ensure_ascii=False)
            log(f"      [Pass 0] Cached → {clean_cache_path}")

    # --- Pass 1: Identify speakers ---
    # Treat an empty cached map as a cache miss — empty means the previous run
    # failed to parse the LLM response and must be retried.
    _speaker_cache_valid = (
        speaker_cache_path is not None
        and speaker_cache_path.exists()
    )
    if _speaker_cache_valid:
        with open(speaker_cache_path, encoding="utf-8") as _fh:  # type: ignore[arg-type]
            speaker_map = json.load(_fh)
        if speaker_map:
            log(f"      [Pass 1] Loaded {len(speaker_map)} speakers from cache.")
        else:
            log("      [Pass 1] Cached speaker map is empty — re-running identification…")
            speaker_map = identify_speakers(cleaned, backend, model, gemini_api_key)
            log(f"      [Pass 1] Identified {len(speaker_map)} speakers.")
            with open(speaker_cache_path, "w", encoding="utf-8") as _fh:  # type: ignore[arg-type]
                json.dump(speaker_map, _fh, indent=2, ensure_ascii=False)
    else:
        log(f"      [Pass 1] Identifying speakers with {backend}/{model}…")
        speaker_map = identify_speakers(cleaned, backend, model, gemini_api_key)
        log(f"      [Pass 1] Identified {len(speaker_map)} speakers.")
        if speaker_cache_path is not None:
            speaker_cache_path.parent.mkdir(parents=True, exist_ok=True)
            with open(speaker_cache_path, "w", encoding="utf-8") as _fh:
                json.dump(speaker_map, _fh, indent=2, ensure_ascii=False)
            log(f"      [Pass 1] Cached → {speaker_cache_path}")

    # --- Pass 2: Extract catalogue ---
    if catalogue_cache_path is not None and catalogue_cache_path.exists():
        log(f"      [Pass 2] Loading cached catalogue from {catalogue_cache_path}…")
        with open(catalogue_cache_path, encoding="utf-8") as _fh:
            catalogue = json.load(_fh)
        log("      [Pass 2] Catalogue loaded.")
    else:
        n_chunks = -(-len(cleaned) // chunk_size)  # ceiling division
        log(f"      [Pass 2] Extracting catalogue ({n_chunks} chunks × {chunk_size} segs) with {backend}/{model}…")
        catalogue = extract_catalogue(cleaned, speaker_map, backend, model, gemini_api_key, chunk_size)
        log(f"      [Pass 2] Catalogue: {len(catalogue['pcs'])} PCs, {len(catalogue['npcs'])} NPCs, "
            f"{len(catalogue['locations'])} locations, {len(catalogue['enemies'])} enemies.")
        if catalogue_cache_path is not None:
            catalogue_cache_path.parent.mkdir(parents=True, exist_ok=True)
            with open(catalogue_cache_path, "w", encoding="utf-8") as _fh:
                json.dump(catalogue, _fh, indent=2, ensure_ascii=False)
            log(f"      [Pass 2] Cached → {catalogue_cache_path}")

    # --- Pass 3: Generate outline ---
    if outline_cache_path is not None and outline_cache_path.exists():
        log(f"      [Pass 3] Loading cached outline from {outline_cache_path}…")
        with open(outline_cache_path, encoding="utf-8") as _fh:
            outline = _fh.read()
        log("      [Pass 3] Outline loaded.")
    else:
        n_chunks = -(-len(cleaned) // chunk_size)
        log(f"      [Pass 3] Generating outline ({n_chunks} chunks) with {backend}/{model}…")
        outline = generate_outline(cleaned, speaker_map, catalogue, backend, model, gemini_api_key, chunk_size)
        log("      [Pass 3] Outline generated.")
        if outline_cache_path is not None:
            outline_cache_path.parent.mkdir(parents=True, exist_ok=True)
            with open(outline_cache_path, "w", encoding="utf-8") as _fh:
                _fh.write(outline)
            log(f"      [Pass 3] Cached → {outline_cache_path}")

    # --- Pass 3.5: Compress outline to fit narrative context window ---
    if compressed_outline_cache_path is not None and compressed_outline_cache_path.exists():
        log(f"      [Pass 3.5] Loading cached compressed outline…")
        with open(compressed_outline_cache_path, encoding="utf-8") as _fh:
            compressed = _fh.read()
        log("      [Pass 3.5] Compressed outline loaded.")
    else:
        log(f"      [Pass 3.5] Compressing outline with {backend}/{model}…")
        compressed = compress_outline(outline, backend, model, gemini_api_key)
        log(f"      [Pass 3.5] Compressed: {len(outline)} → {len(compressed)} chars.")
        if compressed_outline_cache_path is not None:
            compressed_outline_cache_path.parent.mkdir(parents=True, exist_ok=True)
            with open(compressed_outline_cache_path, "w", encoding="utf-8") as _fh:
                _fh.write(compressed)
            log(f"      [Pass 3.5] Cached → {compressed_outline_cache_path}")

    # --- Pass 4: Generate narrative ---
    log(f"      [Pass 4] Writing final recap with {backend}/{model}…")
    narrative = generate_narrative(catalogue, compressed, backend, model, gemini_api_key)
    log("      [Pass 4] Recap generated.")

    # --- Pass 4.5: Continuity check ---
    log(f"      [Pass 4.5] Running continuity check with {backend}/{model}…")
    narrative = continuity_check(narrative, compressed, backend, model, gemini_api_key)
    log("      [Pass 4.5] Continuity check complete.")

    return save_narrative(narrative, output_path)

