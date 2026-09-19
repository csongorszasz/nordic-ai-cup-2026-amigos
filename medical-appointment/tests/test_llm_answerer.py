"""LLM answerer with a stub client (no model)."""

from answerers import get_answerer
from answerers.llm import LLMAnswerer
from answerers.llm_client import StubClient


def test_chat_template_kwargs_defaults_to_empty(monkeypatch):
    from answerers import llm_client

    monkeypatch.delenv("MEDAPP_LLM_ENABLE_THINKING", raising=False)
    monkeypatch.delenv("MEDAPP_LLM_REASONING_EFFORT", raising=False)
    assert llm_client.chat_template_kwargs() == {}


def test_chat_template_kwargs_thinking_toggle(monkeypatch):
    from answerers import llm_client

    monkeypatch.setenv("MEDAPP_LLM_ENABLE_THINKING", "0")
    assert llm_client.chat_template_kwargs() == {"enable_thinking": False}
    monkeypatch.setenv("MEDAPP_LLM_ENABLE_THINKING", "1")
    monkeypatch.setenv("MEDAPP_LLM_REASONING_EFFORT", "low")
    assert llm_client.chat_template_kwargs() == {
        "enable_thinking": True, "reasoning_effort": "low",
    }

WORDS = [
    {"word": " The", "start": 1.5, "end": 1.7},
    {"word": " dose", "start": 1.8, "end": 2.0},
    {"word": " is", "start": 2.1, "end": 2.2},
    {"word": " 100", "start": 2.3, "end": 2.5},
    {"word": " mg.", "start": 2.6, "end": 2.9},
]
TRANSCRIPT = {
    "segments": [{"id": 0, "start": 1.5, "end": 2.9, "text": "The dose is 100 mg."}],
    "words": WORDS,
}


def test_yes_with_quote():
    client = StubClient([
        '{"answers":[{"id":"q01","answer":"yes","evidence_quote":"The dose is 100 mg."}]}'
    ])
    answerer = LLMAnswerer(client=client, few_shot=())
    assert answerer.answer_all(["Was the dose 100 mg?"], TRANSCRIPT) == [(True, (1.5, 2.9))]


def test_no_answer():
    client = StubClient([
        '{"answers":[{"id":"q01","answer":"no","evidence_quote":null}]}'
    ])
    assert LLMAnswerer(client=client, few_shot=()).answer_all(["Q?"], TRANSCRIPT) == [(False, None)]


def test_parse_failure_is_no():
    assert LLMAnswerer(client=StubClient(["garbage"]), few_shot=()).answer_all(
        ["Q?"], TRANSCRIPT
    ) == [(False, None)]


def test_yes_with_unfindable_quote_becomes_no():
    client = StubClient([
        '{"answers":[{"id":"q01","answer":"yes","evidence_quote":"one hundred milligrams"}]}'
    ])
    assert LLMAnswerer(client=client, few_shot=()).answer_all(["Q?"], TRANSCRIPT) == [(False, None)]


def test_empty_transcript():
    client = StubClient(["{}"])
    assert LLMAnswerer(client=client, few_shot=()).answer_all(
        ["Q?"], {"words": [], "segments": []}
    ) == [(False, None)]


def test_factory_builds_llm():
    assert get_answerer("llm").name == "llm"


def test_scoped_quote_uses_the_cited_occurrence():
    words = [
        {**word, "seg_idx": 0} for word in WORDS
    ] + [
        {**word, "start": word["start"] + 10, "end": word["end"] + 10, "seg_idx": 1}
        for word in WORDS
    ]
    transcript = {
        "words": words,
        "segments": [
            TRANSCRIPT["segments"][0],
            {"id": 1, "start": 11.5, "end": 12.9, "text": "The dose is 100 mg."},
        ],
    }
    client = StubClient([
        '{"answers":[{"id":"q01","answer":"yes","evidence_quote":"The dose is 100 mg.",'
        '"segment_start":"s01","segment_end":"s01"}]}'
    ])
    answerer = LLMAnswerer(client=client, few_shot=(), variant="scoped")
    assert answerer.answer_all(["Was the dose 100 mg?"], transcript) == [(True, (11.5, 12.9))]


def test_scoped_quote_cannot_fall_back_to_global_search():
    client = StubClient([
        '{"answers":[{"id":"q01","answer":"yes","evidence_quote":"The dose is 100 mg.",'
        '"segment_start":"s99","segment_end":"s99"}]}'
    ])
    answerer = LLMAnswerer(client=client, few_shot=(), variant="scoped")
    assert answerer.answer_all(["Was the dose 100 mg?"], TRANSCRIPT) == [(False, None)]


def test_measured_boundary_correction_is_optional_and_preserves_decisions():
    import asr
    from answerers.boundaries import OffsetCalibration

    client = StubClient([
        '{"answers":[{"id":"q01","answer":"yes","evidence_quote":"The dose is 100 mg."}]}'
    ])
    client.model_name = "model"
    client.revision = "revision"
    calibration = OffsetCalibration(
        0.2, 0.0, asr.config_hash(), "model", "revision", "base", "hash"
    )
    answerer = LLMAnswerer(client=client, few_shot=(), calibration=calibration)
    assert answerer.answer_all(["Was the dose 100 mg?"], TRANSCRIPT) == [(True, (1.7, 2.9))]


def test_l1_probe_uses_serving_parse_fallback():
    from llm_probe import _decide_and_cite

    rows = [{
        "question_id": "q1", "transcript_id": "s1", "question": "Was the dose 200 mg?",
        "question_type": "hard_negative", "label": "0",
        "evidence_start": "", "evidence_end": "",
    }]
    records, _, failures = _decide_and_cite(
        "L1", StubClient(["broken"]), TRANSCRIPT, rows, ()
    )
    assert failures == 1
    assert records[0]["prediction"] == 0
    assert records[0]["span"] is None


def test_unknown_prompt_is_rejected_before_model_loading():
    import pytest

    with pytest.raises(ValueError, match="Unknown LLM prompt"):
        LLMAnswerer(client=StubClient(["{}"]), few_shot=(), variant="unknown")


def test_nonpositive_generation_budget_is_rejected_without_loading_weights():
    import pytest
    from answerers.llm_client import HFClient

    with pytest.raises(ValueError, match="positive integer"):
        HFClient(max_new_tokens=0)


def test_beam_width_is_explicit_and_defaults_to_greedy(monkeypatch):
    import pytest
    from answerers.llm_client import HFClient

    monkeypatch.delenv("MEDAPP_LLM_NUM_BEAMS", raising=False)
    assert HFClient().num_beams == 1
    assert HFClient(num_beams=2).num_beams == 2
    with pytest.raises(ValueError, match="positive integer"):
        HFClient(num_beams=0)
