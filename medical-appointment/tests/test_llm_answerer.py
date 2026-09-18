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


def test_parse_failure_is_no():
    assert LLMAnswerer(client=StubClient(["garbage"]), few_shot=()).answer_all(
        ["Q?"], TRANSCRIPT
    ) == [(False, None)]


def test_yes_with_unfindable_quote_has_no_span():
    client = StubClient([
        '{"answers":[{"id":"q01","answer":"yes","evidence_quote":"one hundred milligrams"}]}'
    ])
    assert LLMAnswerer(client=client, few_shot=()).answer_all(["Q?"], TRANSCRIPT) == [(True, None)]


def test_empty_transcript():
    client = StubClient(["{}"])
    assert LLMAnswerer(client=client, few_shot=()).answer_all(
        ["Q?"], {"words": [], "segments": []}
    ) == [(False, None)]


def test_factory_builds_llm():
    assert get_answerer("llm").name == "llm"


class FakeRefiner:
    def __init__(self):
        self.calls = []

    def refine(self, question, words, first_word, last_word):
        self.calls.append((question, first_word, last_word))
        return (1.6, 2.4)


class ExplodingRefiner:
    def refine(self, *args):
        raise RuntimeError("boom")


def test_refiner_replaces_llm_span():
    client = StubClient([
        '{"answers":[{"id":"q01","answer":"yes","evidence_quote":"The dose is 100 mg."}]}'
    ])
    refiner = FakeRefiner()
    answerer = LLMAnswerer(client=client, few_shot=(), refiner=refiner)
    result = answerer.answer_all(["Was the dose 100 mg?"], TRANSCRIPT, return_info=True)
    assert result == [(True, (1.6, 2.4), {
        "decided_by": "yes",
        "quote": "The dose is 100 mg.",
        "llm_span": [1.5, 2.9],
    })]
    assert refiner.calls == [("Was the dose 100 mg?", 0, 4)]


def test_refiner_failure_falls_back_to_llm_span():
    client = StubClient([
        '{"answers":[{"id":"q01","answer":"yes","evidence_quote":"The dose is 100 mg."}]}'
    ])
    answerer = LLMAnswerer(client=client, few_shot=(), refiner=ExplodingRefiner())
    assert answerer.answer_all(["Q?"], TRANSCRIPT) == [(True, (1.5, 2.9))]
