"""LLM answerer with a stub client (no model)."""

from answerers import get_answerer
from answerers.llm import LLMAnswerer
from answerers.llm_client import StubClient

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


def test_parse_failure_falls_back_to_the_prior_guess():
    answerer = LLMAnswerer(
        client=StubClient(["garbage"]), few_shot=(), fallback=lambda qs: [False] * len(qs)
    )
    assert answerer.answer_all(["Q?"], TRANSCRIPT) == [(False, None)]
    answerer = LLMAnswerer(
        client=StubClient(["garbage"]), few_shot=(), fallback=lambda qs: [True] * len(qs)
    )
    assert answerer.answer_all(["Q?"], TRANSCRIPT) == [(True, None)]


def test_yes_with_unfindable_quote_has_no_span():
    client = StubClient([
        '{"answers":[{"id":"q01","answer":"yes","evidence_quote":"one hundred milligrams"}]}'
    ])
    assert LLMAnswerer(client=client, few_shot=()).answer_all(["Q?"], TRANSCRIPT) == [(True, None)]


def test_empty_transcript():
    client = StubClient(["{}"])
    assert LLMAnswerer(
        client=client, few_shot=(), fallback=lambda qs: [False] * len(qs)
    ).answer_all(["Q?"], {"words": [], "segments": []}) == [(False, None)]
    assert client.calls == []


def test_factory_builds_llm():
    assert get_answerer("llm").name == "llm"
