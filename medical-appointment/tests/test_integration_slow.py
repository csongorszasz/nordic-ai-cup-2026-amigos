"""Slow, model-backed integration checks. Run on IDUN: pytest -m slow."""

import pytest

from utils import load_sample_audio


@pytest.mark.slow
def test_transcribe_and_window_one_sample():
    import asr
    import windows as windows_module

    audio_filename = "conversation_sample_17.mp3"
    transcript = asr.transcribe_bytes(
        load_sample_audio(audio_filename), audio_filename, cache=True
    )

    assert transcript["words"], "no words returned"
    assert transcript["duration"] > 60
    assert transcript["language"] == "en"

    wins = windows_module.windows_from_transcript(transcript)
    assert wins
    assert wins[0].first_word == 0
    assert wins[-1].last_word == len(transcript["words"]) - 1
