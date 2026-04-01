"""Tests for ttrpg_narrator.writer — Phase 3 pipeline."""

import json
from unittest.mock import patch, call

from ttrpg_narrator.writer import (
    clean_table_talk,
    identify_speakers,
    extract_catalogue,
    generate_outline,
    generate_narrative,
)


SAMPLE_SEGMENTS = [
    {"start": 0.0,  "end": 5.0,  "speaker": "SPEAKER_00", "text": "I draw my sword and charge the goblin."},
    {"start": 5.5,  "end": 10.0, "speaker": "SPEAKER_01", "text": "The goblin snarls and raises its club."},
    {"start": 10.5, "end": 14.0, "speaker": "SPEAKER_00", "text": "I roll for attack. What's the AC?"},
    {"start": 14.5, "end": 18.0, "speaker": "SPEAKER_01", "text": "The GM describes a dark corridor."},
    {"start": 18.5, "end": 22.0, "speaker": "SPEAKER_00", "text": "We enter cautiously."},
]

SAMPLE_SPEAKER_MAP = {
    "SPEAKER_00": {"role": "player", "character": "Asha"},
    "SPEAKER_01": {"role": "GM"},
}

SAMPLE_CATALOGUE = {
    "pcs": [{"name": "Asha", "speaker": "SPEAKER_00"}],
    "npcs": [{"name": "Goblin", "notes": "enemy"}],
    "locations": [{"name": "Dark corridor", "scale": "micro"}],
    "items": [],
    "enemies": [{"name": "Goblin"}],
    "puzzles": [],
    "quotes": [],
    "quest_goal": "Escape the dungeon",
    "setting": "A dungeon",
    "key_decisions": [],
}


# ---------------------------------------------------------------------------
# Existing tests — clean_table_talk
# ---------------------------------------------------------------------------

class TestCleanTableTalk:
    """Tests for clean_table_talk robustness."""

    def test_falls_back_to_original_when_llm_returns_empty_list(self):
        with patch("ttrpg_narrator.writer._llm_call", return_value="[]"):
            result = clean_table_talk(SAMPLE_SEGMENTS, backend="ollama", model="llama3")
        assert result == SAMPLE_SEGMENTS

    def test_returns_llm_filtered_segments_when_non_empty(self):
        filtered = [SAMPLE_SEGMENTS[0], SAMPLE_SEGMENTS[1]]
        with patch("ttrpg_narrator.writer._llm_call", return_value=json.dumps(filtered)):
            result = clean_table_talk(SAMPLE_SEGMENTS, backend="ollama", model="llama3")
        assert result == filtered

    def test_falls_back_to_original_on_invalid_json(self):
        with patch("ttrpg_narrator.writer._llm_call", return_value="not json"):
            result = clean_table_talk(SAMPLE_SEGMENTS, backend="ollama", model="llama3")
        assert result == SAMPLE_SEGMENTS

    def test_falls_back_to_original_on_non_list_json(self):
        with patch("ttrpg_narrator.writer._llm_call", return_value='{"error": "oops"}'):
            result = clean_table_talk(SAMPLE_SEGMENTS, backend="ollama", model="llama3")
        assert result == SAMPLE_SEGMENTS


# ---------------------------------------------------------------------------
# Pass 1 — identify_speakers
# ---------------------------------------------------------------------------

class TestIdentifySpeakers:

    def test_only_sends_head_of_transcript(self):
        """Only the first `head_size` segments are sent to the LLM.

        Sending the entire transcript wastes context.  A 150-line sample is
        enough to identify which speaker is the GM by their descriptive voice.
        """
        captured = {}

        def fake_llm(system, user, backend, model, gemini_api_key=None):
            captured["user"] = user
            return json.dumps(SAMPLE_SPEAKER_MAP)

        # Pass 10 segments but head_size=3 — only 3 should be in the user content
        with patch("ttrpg_narrator.writer._llm_call", side_effect=fake_llm):
            identify_speakers(SAMPLE_SEGMENTS, backend="ollama", model="llama3", head_size=3)

        # With head_size=3, segment 4 and 5 text must not appear
        assert "We enter cautiously" not in captured["user"], (
            "identify_speakers sent more than head_size segments to the LLM."
        )

    def test_returns_speaker_map_dict(self):
        """Returns a dict keyed by SPEAKER_XX with at least a 'role' field."""
        speaker_json = json.dumps(SAMPLE_SPEAKER_MAP)
        with patch("ttrpg_narrator.writer._llm_call", return_value=speaker_json):
            result = identify_speakers(SAMPLE_SEGMENTS, backend="ollama", model="llama3")
        assert isinstance(result, dict)
        assert "SPEAKER_00" in result
        assert result["SPEAKER_00"]["role"] == "player"
        assert result["SPEAKER_01"]["role"] == "GM"

    def test_returns_empty_dict_on_invalid_json(self):
        """Bad LLM response degrades gracefully to an empty map."""
        with patch("ttrpg_narrator.writer._llm_call", return_value="not json"):
            result = identify_speakers(SAMPLE_SEGMENTS, backend="ollama", model="llama3")
        assert result == {}

    def test_returns_empty_dict_on_non_dict_json(self):
        """LLM returning a list instead of a dict degrades gracefully."""
        with patch("ttrpg_narrator.writer._llm_call", return_value="[]"):
            result = identify_speakers(SAMPLE_SEGMENTS, backend="ollama", model="llama3")
        assert result == {}


# ---------------------------------------------------------------------------
# Pass 2 — extract_catalogue
# ---------------------------------------------------------------------------

class TestExtractCatalogue:

    def test_chunks_long_transcript(self):
        """With 5 segments and chunk_size=2, the LLM is called 3 times (2+2+1)."""
        chunk_result = json.dumps({"pcs": [], "npcs": [], "locations": [], "items": [],
                                   "enemies": [], "puzzles": [], "quotes": [],
                                   "quest_goal": "", "setting": "", "key_decisions": []})
        call_count = {"n": 0}

        def fake_llm(system, user, backend, model, gemini_api_key=None):
            call_count["n"] += 1
            return chunk_result

        with patch("ttrpg_narrator.writer._llm_call", side_effect=fake_llm):
            extract_catalogue(
                SAMPLE_SEGMENTS, SAMPLE_SPEAKER_MAP,
                backend="ollama", model="llama3", chunk_size=2
            )

        assert call_count["n"] == 3, (
            f"Expected 3 LLM calls for 5 segments with chunk_size=2, got {call_count['n']}"
        )

    def test_single_chunk_for_short_transcript(self):
        """Transcript shorter than chunk_size results in exactly 1 LLM call."""
        chunk_result = json.dumps({"pcs": [], "npcs": [], "locations": [], "items": [],
                                   "enemies": [], "puzzles": [], "quotes": [],
                                   "quest_goal": "", "setting": "", "key_decisions": []})
        call_count = {"n": 0}

        def fake_llm(system, user, backend, model, gemini_api_key=None):
            call_count["n"] += 1
            return chunk_result

        with patch("ttrpg_narrator.writer._llm_call", side_effect=fake_llm):
            extract_catalogue(
                SAMPLE_SEGMENTS, SAMPLE_SPEAKER_MAP,
                backend="ollama", model="llama3", chunk_size=100
            )

        assert call_count["n"] == 1

    def test_merges_pcs_from_multiple_chunks(self):
        """PCs found in different chunks are combined in the final catalogue."""
        responses = [
            json.dumps({"pcs": [{"name": "Asha"}], "npcs": [], "locations": [],
                        "items": [], "enemies": [], "puzzles": [], "quotes": [],
                        "quest_goal": "", "setting": "", "key_decisions": []}),
            json.dumps({"pcs": [{"name": "Bryn"}], "npcs": [], "locations": [],
                        "items": [], "enemies": [], "puzzles": [], "quotes": [],
                        "quest_goal": "Find the sword", "setting": "A dungeon",
                        "key_decisions": []}),
        ]
        idx = {"i": 0}

        def fake_llm(system, user, backend, model, gemini_api_key=None):
            r = responses[idx["i"]]
            idx["i"] += 1
            return r

        with patch("ttrpg_narrator.writer._llm_call", side_effect=fake_llm):
            result = extract_catalogue(
                SAMPLE_SEGMENTS[:4], SAMPLE_SPEAKER_MAP,
                backend="ollama", model="llama3", chunk_size=2
            )

        pc_names = [pc["name"] for pc in result["pcs"]]
        assert "Asha" in pc_names
        assert "Bryn" in pc_names

    def test_merges_quest_goal_keeping_first_nonempty(self):
        """The first non-empty quest_goal from any chunk is kept."""
        responses = [
            json.dumps({"pcs": [], "npcs": [], "locations": [], "items": [],
                        "enemies": [], "puzzles": [], "quotes": [],
                        "quest_goal": "Find the crown", "setting": "", "key_decisions": []}),
            json.dumps({"pcs": [], "npcs": [], "locations": [], "items": [],
                        "enemies": [], "puzzles": [], "quotes": [],
                        "quest_goal": "Something else", "setting": "", "key_decisions": []}),
        ]
        idx = {"i": 0}

        def fake_llm(system, user, backend, model, gemini_api_key=None):
            r = responses[idx["i"]]
            idx["i"] += 1
            return r

        with patch("ttrpg_narrator.writer._llm_call", side_effect=fake_llm):
            result = extract_catalogue(
                SAMPLE_SEGMENTS[:4], SAMPLE_SPEAKER_MAP,
                backend="ollama", model="llama3", chunk_size=2
            )

        assert result["quest_goal"] == "Find the crown"

    def test_includes_speaker_context_in_prompt(self):
        """The speaker map is included in each chunk call so the LLM knows who's the GM."""
        chunk_result = json.dumps({"pcs": [], "npcs": [], "locations": [], "items": [],
                                   "enemies": [], "puzzles": [], "quotes": [],
                                   "quest_goal": "", "setting": "", "key_decisions": []})
        captured = {}

        def fake_llm(system, user, backend, model, gemini_api_key=None):
            captured["system"] = system
            return chunk_result

        with patch("ttrpg_narrator.writer._llm_call", side_effect=fake_llm):
            extract_catalogue(
                SAMPLE_SEGMENTS[:2], SAMPLE_SPEAKER_MAP,
                backend="ollama", model="llama3", chunk_size=100
            )

        # SPEAKER_01 is the GM — that info must be in the prompt
        assert "SPEAKER_01" in captured["system"] or "GM" in captured["system"], (
            "Speaker map must be included in the catalogue extraction prompt."
        )

    def test_returns_empty_catalogue_keys_on_total_failure(self):
        """If all LLM calls produce unparseable JSON, return an empty catalogue."""
        with patch("ttrpg_narrator.writer._llm_call", return_value="not json"):
            result = extract_catalogue(
                SAMPLE_SEGMENTS, SAMPLE_SPEAKER_MAP,
                backend="ollama", model="llama3", chunk_size=100
            )
        assert "pcs" in result
        assert isinstance(result["pcs"], list)


# ---------------------------------------------------------------------------
# Pass 3 — generate_outline
# ---------------------------------------------------------------------------

class TestGenerateOutline:

    def test_chunks_transcript_and_concatenates(self):
        """With 5 segments and chunk_size=2, three chunk summaries are joined."""
        summaries = ["Scene 1.", "Scene 2.", "Scene 3."]
        idx = {"i": 0}

        def fake_llm(system, user, backend, model, gemini_api_key=None):
            r = summaries[idx["i"]]
            idx["i"] = min(idx["i"] + 1, len(summaries) - 1)
            return r

        with patch("ttrpg_narrator.writer._llm_call", side_effect=fake_llm):
            result = generate_outline(
                SAMPLE_SEGMENTS, SAMPLE_SPEAKER_MAP, SAMPLE_CATALOGUE,
                backend="ollama", model="llama3", chunk_size=2
            )

        assert "Scene 1." in result
        assert "Scene 2." in result
        assert "Scene 3." in result

    def test_includes_catalogue_in_each_chunk_prompt(self):
        """The catalogue context (NPCs, locations…) must be visible in each chunk call."""
        captured = {}

        def fake_llm(system, user, backend, model, gemini_api_key=None):
            captured.setdefault("systems", []).append(system)
            return "A scene."

        with patch("ttrpg_narrator.writer._llm_call", side_effect=fake_llm):
            generate_outline(
                SAMPLE_SEGMENTS[:2], SAMPLE_SPEAKER_MAP, SAMPLE_CATALOGUE,
                backend="ollama", model="llama3", chunk_size=100
            )

        combined = " ".join(captured["systems"])
        # "Goblin" is in the catalogue enemies — it must appear in the prompt context
        assert "Goblin" in combined or "Asha" in combined, (
            "Catalogue context must be included in generate_outline prompt."
        )

    def test_returns_string(self):
        """generate_outline always returns a string."""
        with patch("ttrpg_narrator.writer._llm_call", return_value="A scene."):
            result = generate_outline(
                SAMPLE_SEGMENTS, SAMPLE_SPEAKER_MAP, SAMPLE_CATALOGUE,
                backend="ollama", model="llama3", chunk_size=100
            )
        assert isinstance(result, str)
        assert len(result) > 0


# ---------------------------------------------------------------------------
# Pass 4 — generate_narrative (updated signature)
# ---------------------------------------------------------------------------

class TestGenerateNarrative:

    def test_uses_catalogue_and_outline_not_raw_segments(self):
        """generate_narrative takes catalogue + outline, not a segments list.

        The old approach sent all segments to the LLM and got a thin summary
        because most were truncated.  Using the pre-built outline and catalogue
        means the input is always small and fully grounded.
        """
        captured = {}

        def fake_llm(system, user, backend, model, gemini_api_key=None):
            captured["user"] = user
            return "Session recap."

        with patch("ttrpg_narrator.writer._llm_call", side_effect=fake_llm):
            result = generate_narrative(
                SAMPLE_CATALOGUE, "Scene 1.\nScene 2.",
                backend="ollama", model="llama3",
            )

        assert result == "Session recap."
        # Catalogue content should appear in the prompt
        assert "Goblin" in captured["user"] or "Asha" in captured["user"] or "Scene 1" in captured["user"]

    def test_system_prompt_requests_factual_summary(self):
        """The system prompt asks for a factual recap, not engaging fiction."""
        captured = {}

        def fake_llm(system, user, backend, model, gemini_api_key=None):
            captured["system"] = system
            return "Session recap."

        with patch("ttrpg_narrator.writer._llm_call", side_effect=fake_llm):
            generate_narrative(
                SAMPLE_CATALOGUE, "Scene 1.",
                backend="ollama", model="llama3",
            )

        system = captured["system"].lower()
        assert "summary" in system or "recap" in system or "summarize" in system
        assert "richly written" not in system



class TestCleanTableTalk:
    """Tests for clean_table_talk robustness."""

    def test_falls_back_to_original_when_llm_returns_empty_list(self):
        """If the LLM filters out ALL segments, the original transcript is kept.

        This prevents the storyteller receiving [] and writing generic filler.
        """
        with patch("ttrpg_narrator.writer._llm_call", return_value="[]"):
            result = clean_table_talk(SAMPLE_SEGMENTS, backend="ollama", model="llama3")
        assert result == SAMPLE_SEGMENTS, (
            "clean_table_talk should fall back to the original segments "
            "when the LLM returns an empty list, not produce a blank transcript."
        )

    def test_returns_llm_filtered_segments_when_non_empty(self):
        """Normal path: LLM keeps a subset — that filtered list is returned."""
        filtered = [SAMPLE_SEGMENTS[0], SAMPLE_SEGMENTS[1]]
        import json
        with patch("ttrpg_narrator.writer._llm_call", return_value=json.dumps(filtered)):
            result = clean_table_talk(SAMPLE_SEGMENTS, backend="ollama", model="llama3")
        assert result == filtered

    def test_falls_back_to_original_on_invalid_json(self):
        """Existing behaviour: unparseable response → return original segments."""
        with patch("ttrpg_narrator.writer._llm_call", return_value="not json"):
            result = clean_table_talk(SAMPLE_SEGMENTS, backend="ollama", model="llama3")
        assert result == SAMPLE_SEGMENTS

    def test_falls_back_to_original_on_non_list_json(self):
        """Existing behaviour: LLM returns a dict instead of list → return original."""
        with patch("ttrpg_narrator.writer._llm_call", return_value='{"error": "oops"}'):
            result = clean_table_talk(SAMPLE_SEGMENTS, backend="ollama", model="llama3")
        assert result == SAMPLE_SEGMENTS
