"""Tests for ttrpg_narrator.writer — Phase 3 pipeline."""

import json
from unittest.mock import patch, call

from ttrpg_narrator.writer import (
    clean_table_talk,
    identify_speakers,
    extract_catalogue,
    generate_outline,
    generate_narrative,
    normalize_transcript,
    compress_outline,
    continuity_check,
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

    def test_system_prompt_requests_structured_sections(self):
        """The final recap prompt must explicitly request formatted catalogue sections.

        SampleOutput.txt shows the desired format: narrative paragraphs followed
        by OUTLINE, NPCs, ITEMS, LOCATIONS, QUESTS, and PLAYER CHARACTERS sections.
        """
        captured = {}

        def fake_llm(system, user, backend, model, gemini_api_key=None):
            captured["system"] = system
            return "recap"

        with patch("ttrpg_narrator.writer._llm_call", side_effect=fake_llm):
            generate_narrative(SAMPLE_CATALOGUE, "Scene 1.", backend="ollama", model="llama3")

        system = captured["system"].upper()
        assert "NPC" in system or "CHARACTERS" in system, (
            "System prompt must request an NPCs / Characters section."
        )
        assert "ITEM" in system, "System prompt must request an Items section."
        assert "LOCATION" in system, "System prompt must request a Locations section."
        assert "QUEST" in system or "OUTLINE" in system, (
            "System prompt must request a Quests or Outline section."
        )

    def test_system_prompt_prohibits_meta_game_references(self):
        """The storyteller prompt must explicitly forbid GM/player/dice references.

        The previous prompt produced 'The GM reminded players...' style output.
        The prompt must contain an explicit instruction not to reference real-world
        game elements.
        """
        captured = {}

        def fake_llm(system, user, backend, model, gemini_api_key=None):
            captured["system"] = system
            return "recap"

        with patch("ttrpg_narrator.writer._llm_call", side_effect=fake_llm):
            generate_narrative(SAMPLE_CATALOGUE, "Scene 1.", backend="ollama", model="llama3")

        system = captured["system"].lower()
        # The prompt must explicitly prohibit meta-game language
        assert "game master" in system or "gm" in system or "dice" in system or "in-universe" in system, (
            "Prompt must mention meta-game elements so it can prohibit them."
        )
        assert "never" in system or "do not" in system or "avoid" in system, (
            "Prompt must contain a prohibition (never/do not/avoid) against meta-game references."
        )

    def test_outline_prompt_avoids_meta_game_fallback_language(self):
        """The outline prompt must not instruct the LLM to use 'a player' as a fallback.

        'Otherwise use \"a player\"' causes the model to write 'a player charged
        the goblin', which is out-of-universe.  The fallback should be in-universe
        language such as 'the party' or 'the adventurers'.
        """
        from ttrpg_narrator.writer import _OUTLINE_SYSTEM
        assert '"a player"' not in _OUTLINE_SYSTEM and "'a player'" not in _OUTLINE_SYSTEM, (
            "_OUTLINE_SYSTEM must not use 'a player' as a fallback label — "
            "use in-universe language like 'the party' or 'the adventurers'."
        )


# ---------------------------------------------------------------------------
# Pass 1 — identify_speakers (new: dynamic coverage)
# ---------------------------------------------------------------------------

class TestIdentifySpeakersCoverage:

    def test_covers_late_appearing_speaker(self):
        """identify_speakers must include ALL unique speakers even if they speak late.

        head_size is a *minimum* context floor. If a speaker only appears PAST
        head_size, the function must extend the head to include their first utterance.

        With head_size=3 and SPEAKER_02 at index 40, the static segments[:3] misses
        SPEAKER_02 entirely. The new implementation must go further.
        """
        early_segs = [
            {"start": float(i), "end": float(i + 1), "speaker": "SPEAKER_00", "text": "some text"}
            for i in range(20)
        ] + [
            {"start": float(i + 20), "end": float(i + 21), "speaker": "SPEAKER_01", "text": "other text"}
            for i in range(20)
        ]
        late_seg = {"start": 1000.0, "end": 1001.0, "speaker": "SPEAKER_02", "text": "I am a late arrival"}
        segments = early_segs + [late_seg]  # 41 segments; SPEAKER_02 is last

        captured = {}

        def fake_llm(system, user, backend, model, gemini_api_key=None):
            captured["user"] = user
            return json.dumps({
                "SPEAKER_00": {"role": "GM"},
                "SPEAKER_01": {"role": "player", "character": "Asha"},
                "SPEAKER_02": {"role": "player", "character": "Bryn"},
            })

        # head_size=3 is the minimum floor — SPEAKER_02 at index 40 must still be included
        with patch("ttrpg_narrator.writer._llm_call", side_effect=fake_llm):
            identify_speakers(segments, backend="ollama", model="llama3", head_size=3)

        assert "I am a late arrival" in captured["user"], (
            "identify_speakers must extend past head_size=3 to include SPEAKER_02 at index 40."
        )

    def test_covers_speaker_past_default_head_size(self):
        """SPEAKER_02 at index 200 must be included even with the default head_size=150.

        The static segments[:150] implementation drops everyone past index 149.
        The dynamic implementation extends until all unique speakers are covered.
        """
        base = [
            {"start": float(i), "end": float(i + 1), "speaker": "SPEAKER_00", "text": "first"}
            for i in range(100)
        ] + [
            {"start": float(i + 100), "end": float(i + 101), "speaker": "SPEAKER_01", "text": "second"}
            for i in range(100)
        ]
        # SPEAKER_02 first appears at index 200 — past the default head_size of 150
        late_seg = {"start": 2000.0, "end": 2001.0, "speaker": "SPEAKER_02", "text": "third speaker here"}
        segments = base + [late_seg]  # 201 segments

        captured = {}

        def fake_llm(system, user, backend, model, gemini_api_key=None):
            captured["user"] = user
            return json.dumps({
                "SPEAKER_00": {"role": "GM"},
                "SPEAKER_01": {"role": "player"},
                "SPEAKER_02": {"role": "player"},
            })

        with patch("ttrpg_narrator.writer._llm_call", side_effect=fake_llm):
            identify_speakers(segments, backend="ollama", model="llama3")  # default head_size=150

        assert "third speaker here" in captured["user"], (
            "identify_speakers must extend past default head_size=150 to include "
            "SPEAKER_02 whose first utterance is at index 200."
        )


# ---------------------------------------------------------------------------
# normalize_transcript
# ---------------------------------------------------------------------------

class TestNormalizeTranscript:

    def test_merges_consecutive_same_speaker(self):
        """Multiple consecutive segments from the same speaker become one entry."""
        segments = [
            {"start": 0.0, "end": 1.0, "speaker": "SPEAKER_01", "text": "Hello there."},
            {"start": 1.5, "end": 2.5, "speaker": "SPEAKER_01", "text": "How are you?"},
            {"start": 3.0, "end": 4.0, "speaker": "SPEAKER_00", "text": "Fine thanks indeed."},
        ]
        result = normalize_transcript(segments)
        assert len(result) == 2
        assert result[0]["speaker"] == "SPEAKER_01"
        assert "Hello there" in result[0]["text"]
        assert "How are you" in result[0]["text"]

    def test_does_not_merge_different_speakers(self):
        """A speaker change produces a new entry even for short lines."""
        segments = [
            {"start": 0.0, "end": 1.0, "speaker": "SPEAKER_01", "text": "Charge the gate!"},
            {"start": 1.5, "end": 2.5, "speaker": "SPEAKER_00", "text": "The gate holds firm."},
            {"start": 3.0, "end": 4.0, "speaker": "SPEAKER_01", "text": "Try again harder."},
        ]
        result = normalize_transcript(segments)
        assert len(result) == 3
        assert result[1]["speaker"] == "SPEAKER_00"

    def test_output_has_no_timestamps(self):
        """Normalized segments must not carry 'start' or 'end' fields."""
        segments = [{"start": 0.0, "end": 1.0, "speaker": "SPEAKER_01", "text": "Hello world."}]
        result = normalize_transcript(segments)
        for entry in result:
            assert "start" not in entry
            assert "end" not in entry

    def test_filters_empty_text_segments(self):
        """Segments with blank or whitespace-only text are dropped silently."""
        segments = [
            {"start": 0.0, "end": 0.3, "speaker": "SPEAKER_01", "text": "   "},
            {"start": 1.0, "end": 3.0, "speaker": "SPEAKER_00", "text": "I draw my sword."},
        ]
        result = normalize_transcript(segments)
        assert len(result) == 1
        assert result[0]["speaker"] == "SPEAKER_00"

    def test_dramatically_reduces_segment_count(self):
        """Merging consecutive runs collapses to one entry per speaker block."""
        segments = (
            [{"start": float(i), "end": float(i+1), "speaker": "SPEAKER_01",
              "text": f"GM says thing {i}."}
             for i in range(20)]
            + [{"start": float(i+20), "end": float(i+21), "speaker": "SPEAKER_00",
                "text": f"Player says {i}."}
               for i in range(5)]
        )
        result = normalize_transcript(segments)
        assert len(result) == 2  # one GM block, one player block


# ---------------------------------------------------------------------------
# identify_speakers — JSON robustness
# ---------------------------------------------------------------------------

class TestIdentifySpeakersJsonRobustness:

    def test_extracts_json_embedded_in_commentary(self):
        """identify_speakers must succeed when the LLM wraps the JSON in prose.

        A common llama3 failure: the model writes commentary before/after the JSON.
        _strip_fences does not handle this, so the result was always {}.
        The new _extract_json_object helper must find and parse the embedded dict.
        """
        noisy_response = (
            "Sure, here is the speaker identification:\n"
            '{"SPEAKER_00": {"role": "player", "character": "Asha"}, '
            '"SPEAKER_01": {"role": "GM"}}'
            "\nLet me know if you need anything else."
        )
        with patch("ttrpg_narrator.writer._llm_call", return_value=noisy_response):
            result = identify_speakers(SAMPLE_SEGMENTS, backend="ollama", model="llama3")
        assert result.get("SPEAKER_01", {}).get("role") == "GM", (
            "identify_speakers must extract JSON even when wrapped in commentary."
        )

    def test_returns_empty_dict_when_no_json_anywhere(self):
        """When there is truly no JSON in the LLM response, return {} gracefully."""
        with patch("ttrpg_narrator.writer._llm_call",
                   return_value="I cannot identify the speakers from this text."):
            result = identify_speakers(SAMPLE_SEGMENTS, backend="ollama", model="llama3")
        assert result == {}


# ---------------------------------------------------------------------------
# compress_outline
# ---------------------------------------------------------------------------

class TestCompressOutline:

    def test_calls_llm_once_for_short_outline(self):
        """A short outline compresses in a single LLM call."""
        short_outline = "- Scene 1.\n- Scene 2.\n- Scene 3."
        call_count = {"n": 0}

        def fake_llm(system, user, backend, model, gemini_api_key=None):
            call_count["n"] += 1
            return "- Compressed scene."

        with patch("ttrpg_narrator.writer._llm_call", side_effect=fake_llm):
            result = compress_outline(short_outline, backend="ollama", model="llama3")

        assert call_count["n"] == 1
        assert isinstance(result, str) and len(result) > 0

    def test_chunks_long_outline(self):
        """60 bullet lines with chunk_size=20 results in 3 LLM calls."""
        long_outline = "\n".join(f"- Event number {i}." for i in range(60))
        call_count = {"n": 0}

        def fake_llm(system, user, backend, model, gemini_api_key=None):
            call_count["n"] += 1
            return "- Compressed event."

        with patch("ttrpg_narrator.writer._llm_call", side_effect=fake_llm):
            compress_outline(long_outline, backend="ollama", model="llama3", chunk_size=20)

        assert call_count["n"] == 3, (
            f"Expected 3 calls for 60 lines / chunk_size=20, got {call_count['n']}"
        )

    def test_combines_all_chunk_results(self):
        """The returned string contains output from every compressed chunk."""
        long_outline = "\n".join(f"- Event {i}." for i in range(40))
        responses = ["- Chunk A summary.", "- Chunk B summary."]
        idx = {"i": 0}

        def fake_llm(system, user, backend, model, gemini_api_key=None):
            r = responses[idx["i"]]
            idx["i"] += 1
            return r

        with patch("ttrpg_narrator.writer._llm_call", side_effect=fake_llm):
            result = compress_outline(
                long_outline, backend="ollama", model="llama3", chunk_size=20
            )

        assert "Chunk A summary" in result
        assert "Chunk B summary" in result


# ---------------------------------------------------------------------------
# continuity_check
# ---------------------------------------------------------------------------

class TestContinuityCheck:

    def test_sends_both_outline_and_narrative_to_llm(self):
        """The LLM receives both the narrative draft and the source outline."""
        captured = {}

        def fake_llm(system, user, backend, model, gemini_api_key=None):
            captured["user"] = user
            return "Corrected narrative."

        with patch("ttrpg_narrator.writer._llm_call", side_effect=fake_llm):
            continuity_check(
                narrative="Children were unharmed.",
                outline="- Nobby killed two of the possessed children.",
                backend="ollama", model="llama3",
            )

        assert "Children were unharmed" in captured["user"]
        assert "Nobby killed two" in captured["user"]

    def test_returns_corrected_narrative_string(self):
        """continuity_check returns whatever the LLM writes as the correction."""
        with patch("ttrpg_narrator.writer._llm_call",
                   return_value="Updated narrative with corrections applied."):
            result = continuity_check(
                narrative="Draft.", outline="- Event.", backend="ollama", model="llama3"
            )
        assert result == "Updated narrative with corrections applied."


# ---------------------------------------------------------------------------
# generate_outline — preamble stripping
# ---------------------------------------------------------------------------

class TestGenerateOutlinePreambleStripping:

    def test_strips_here_are_the_bullet_points_preamble(self):
        """LLM preamble 'Here are the bullet points...' must never appear in output."""
        preamble_response = (
            "Here are the bullet points summarizing what happened in this chunk:\n\n"
            "- Nobby charged the gate.\n"
            "- Asha found a key."
        )
        with patch("ttrpg_narrator.writer._llm_call", return_value=preamble_response):
            result = generate_outline(
                SAMPLE_SEGMENTS, SAMPLE_SPEAKER_MAP, SAMPLE_CATALOGUE,
                backend="ollama", model="llama3", chunk_size=100,
            )
        assert "Here are the bullet points" not in result
        assert "Nobby charged the gate" in result

    def test_strips_this_chunk_covers_preamble(self):
        """'This chunk covers...' variants are also stripped."""
        response = (
            "This chunk covers the following events:\n"
            "- The party entered the cathedral.\n"
            "- They found the children."
        )
        with patch("ttrpg_narrator.writer._llm_call", return_value=response):
            result = generate_outline(
                SAMPLE_SEGMENTS, SAMPLE_SPEAKER_MAP, SAMPLE_CATALOGUE,
                backend="ollama", model="llama3", chunk_size=100,
            )
        assert "This chunk covers" not in result
        assert "The party entered the cathedral" in result


# ---------------------------------------------------------------------------
# Prompt content: dice → narrative and GM narration
# ---------------------------------------------------------------------------

class TestOutlinePromptInstructions:

    def test_outline_prompt_converts_dice_to_narrative_outcomes(self):
        """_OUTLINE_SYSTEM must instruct the LLM to convert dice results to
        story descriptions of what happened, not raw numbers."""
        from ttrpg_narrator.writer import _OUTLINE_SYSTEM
        prompt_lower = _OUTLINE_SYSTEM.lower()
        assert "dice" in prompt_lower or "roll" in prompt_lower, (
            "_OUTLINE_SYSTEM must mention dice/rolls."
        )
        assert "result" in prompt_lower or "outcome" in prompt_lower or "story" in prompt_lower, (
            "_OUTLINE_SYSTEM must say to convert to a story result/outcome."
        )

    def test_outline_prompt_treats_gm_speech_as_world_narration(self):
        """_OUTLINE_SYSTEM must tell the LLM to treat the GM's lines as objective
        world description so scene-setting info is not lost."""
        from ttrpg_narrator.writer import _OUTLINE_SYSTEM
        prompt_lower = _OUTLINE_SYSTEM.lower()
        assert "gm" in prompt_lower or "game master" in prompt_lower or "narrator" in prompt_lower, (
            "_OUTLINE_SYSTEM must reference the GM/narrator role."
        )
