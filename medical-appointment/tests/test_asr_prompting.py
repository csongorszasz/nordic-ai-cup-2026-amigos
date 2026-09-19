"""ASR hints are opt-in, cache-separated, bounded, and passed to actual decoding."""

from types import SimpleNamespace

import pytest

import asr


class Tokenizer:
    def encode(self, text, add_special_tokens):
        assert add_special_tokens is False
        return SimpleNamespace(ids=text.split())


class Model:
    max_length = 448
    hf_tokenizer = Tokenizer()

    def __init__(self):
        self.calls = []

    def transcribe(self, audio, **kwargs):
        self.calls.append(kwargs)
        info = SimpleNamespace(language="en", language_probability=1.0, duration=1.0)
        return iter([]), info


def test_disabled_hints_preserve_published_configuration_hash(monkeypatch):
    monkeypatch.setattr(asr, "MODEL_SIZE", "large-v3-turbo")
    monkeypatch.setattr(asr, "COMPUTE_TYPE", "int8")
    monkeypatch.setattr(asr, "LANGUAGE", "en")
    monkeypatch.setattr(asr, "INITIAL_PROMPT", "")
    monkeypatch.setattr(asr, "HOTWORDS", "")
    assert asr.prompting_options() == {}
    assert asr.config_hash() == "e75a7f6e"


def test_prompt_modes_cannot_share_a_transcript_cache(monkeypatch):
    monkeypatch.setattr(asr, "INITIAL_PROMPT", "")
    monkeypatch.setattr(asr, "HOTWORDS", "")
    original = asr.config_hash()
    monkeypatch.setattr(asr, "INITIAL_PROMPT", "Ibumetin Panodil")
    initial = asr.config_hash()
    monkeypatch.setattr(asr, "INITIAL_PROMPT", "")
    monkeypatch.setattr(asr, "HOTWORDS", "Ibumetin Panodil")
    repeated = asr.config_hash()
    assert len({original, initial, repeated}) == 3


def test_both_hints_share_one_explicit_prompt_budget(monkeypatch):
    monkeypatch.setattr(asr, "INITIAL_PROMPT", "word " * 120)
    monkeypatch.setattr(asr, "HOTWORDS", "term " * 120)
    with pytest.raises(ValueError, match="shared 223-token budget"):
        asr.prompting_options(Model())


@pytest.mark.parametrize("mode", ["initial_prompt", "hotwords"])
def test_enabled_hint_reaches_decoding_and_transcript_metadata(monkeypatch, mode):
    model = Model()
    monkeypatch.setattr(asr, "get_model", lambda: model)
    monkeypatch.setattr(asr, "INITIAL_PROMPT", "Ibumetin Panodil" if mode == "initial_prompt" else "")
    monkeypatch.setattr(asr, "HOTWORDS", "Ibumetin Panodil" if mode == "hotwords" else "")
    result = asr.transcribe_bytes(b"audio", "sample.mp3", cache=False)
    assert model.calls[0][mode] == "Ibumetin Panodil"
    assert model.calls[0]["condition_on_previous_text"] is False
    assert model.calls[0]["word_timestamps"] is True
    assert result["prompting"] == {mode: "Ibumetin Panodil"}


def test_unprompted_output_and_call_shape_are_unchanged(monkeypatch):
    model = Model()
    monkeypatch.setattr(asr, "get_model", lambda: model)
    monkeypatch.setattr(asr, "INITIAL_PROMPT", "")
    monkeypatch.setattr(asr, "HOTWORDS", "")
    result = asr.transcribe_bytes(b"audio", "sample.mp3", cache=False)
    assert "prompting" not in result
    assert model.calls == [{
        "language": asr.LANGUAGE, "word_timestamps": True,
        "vad_filter": True, "condition_on_previous_text": False,
    }]


def test_warmup_validates_the_same_hint_budget(monkeypatch):
    model = Model()
    monkeypatch.setattr(asr, "get_model", lambda: model)
    monkeypatch.setattr(asr, "INITIAL_PROMPT", "")
    monkeypatch.setattr(asr, "HOTWORDS", "term " * 224)
    with pytest.raises(ValueError, match="shorten the vocabulary"):
        asr.warm_up()
    assert model.calls == []
