"""Tests for ttrpg_narrator.writer — Phase 3 pipeline."""

from unittest.mock import patch

from ttrpg_narrator.writer import clean_table_talk, generate_narrative


SAMPLE_SEGMENTS = [
    {"start": 0.0, "end": 5.0, "speaker": "SPEAKER_00", "text": "I draw my sword and charge the goblin."},
    {"start": 5.5, "end": 10.0, "speaker": "SPEAKER_01", "text": "The goblin snarls and raises its club."},
    {"start": 10.5, "end": 14.0, "speaker": "SPEAKER_00", "text": "I roll for attack. What's the AC?"},
]


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


class TestGenerateNarrative:
    def test_passes_transcript_as_plain_text(self):
        """The transcript is passed to the LLM as compact plain text, not JSON.

        Sending 1000+ segments as indented JSON uses 3-4x more tokens than
        plain 'SPEAKER_XX: text' lines, blowing past llama3's context window
        and causing most of the session to be silently truncated.
        """
        import json

        captured = {}

        def fake_llm(system, user, backend, model, gemini_api_key=None):
            captured["user"] = user
            return "The party fought the goblin."

        with patch("ttrpg_narrator.writer._llm_call", side_effect=fake_llm):
            result = generate_narrative(SAMPLE_SEGMENTS, backend="ollama", model="llama3")

        assert result == "The party fought the goblin."
        user_content = captured["user"]
        # Must NOT be JSON — the old behaviour broke large sessions
        try:
            json.loads(user_content)
            is_json = True
        except (json.JSONDecodeError, ValueError):
            is_json = False
        assert not is_json, (
            "generate_narrative should send plain text, not JSON, "
            "to avoid overflowing the model's context window."
        )
        # Must contain all three speaker utterances in readable form
        assert "SPEAKER_00: I draw my sword and charge the goblin." in user_content
        assert "SPEAKER_01: The goblin snarls and raises its club." in user_content
        assert "SPEAKER_00: I roll for attack. What's the AC?" in user_content

    def test_system_prompt_requests_factual_summary(self):
        """The system prompt should ask for a factual session summary, not fiction.

        The old prompt ('richly written... engaging story') produced mystical
        narrator prose. We want a plain recap of what actually happened.
        """
        captured = {}

        def fake_llm(system, user, backend, model, gemini_api_key=None):
            captured["system"] = system
            return "The party fought the goblin."

        with patch("ttrpg_narrator.writer._llm_call", side_effect=fake_llm):
            generate_narrative(SAMPLE_SEGMENTS, backend="ollama", model="llama3")

        system = captured["system"].lower()
        assert "summary" in system or "recap" in system or "summarize" in system, (
            "System prompt should ask for a summary/recap of the session."
        )
        assert "richly written" not in system, (
            "System prompt must not ask for 'richly written' prose — "
            "that produces editorialized fantasy fiction instead of a session log."
        )
