"""Serving hardening: truncation salvage, deadlines, the GPU lock, the prior
fallback, occurrence-aware alignment, span offsets, few-shot and hotwords."""

import base64
import json
import threading
import time

import pytest

import asr
import example
from answerers import llm as llm_module
from answerers.align import align_quote, align_span, apply_offsets, fuzzy_align, segment_at
from answerers.llm import LLMAnswerer
from answerers.llm_client import StubClient
from answerers.llm_parse import parse_answers, salvage_objects
from answerers.llm_prompt import build_few_shot, build_l1_messages, qid_for
from dtos import ASRQuestionRequestDto
from prior import QuestionPrior, guess
from utils import validate_response

IDS = [qid_for(i) for i in range(10)]


def _words(text, seg_every=None):
    words, t = [], 0.0
    for index, token in enumerate(text.split()):
        seg = index // seg_every if seg_every else 0
        words.append({"word": " " + token, "start": t, "end": t + 0.3, "seg_idx": seg})
        t += 0.4
    return words


# --- parsing -----------------------------------------------------------------

def _full_reply(n=10):
    return json.dumps({"answers": [
        {"id": qid, "answer": "yes", "evidence_quote": "take one tablet twice a day"}
        for qid in IDS[:n]
    ]})


def test_truncated_reply_keeps_every_closed_answer():
    reply = _full_reply()
    cut = reply[: reply.index('"q08"') + 3]  # cut inside q08's object
    parsed = parse_answers(cut, IDS)
    assert [parsed[q]["answer"] for q in IDS[:7]] == [True] * 7
    assert all(parsed[q] is None for q in IDS[7:])


def test_complete_reply_unchanged_by_salvage():
    parsed = parse_answers(_full_reply(), IDS)
    assert all(parsed[q]["answer"] is True for q in IDS)


def test_salvage_ignores_braces_inside_strings():
    text = '{"answers":[{"id":"q01","answer":"yes","evidence_quote":"a } b {"},{"id":"q02"'
    assert salvage_objects(text) == [
        {"id": "q01", "answer": "yes", "evidence_quote": "a } b {"}
    ]


@pytest.mark.parametrize("value,expected", [("s07", 7), ("7", 7), (3, 3), ("[s12]", 12), ("x", None)])
def test_segment_field_parsed(value, expected):
    text = json.dumps({"answers": [{"id": "q01", "answer": "yes", "segment": value}]})
    assert parse_answers(text, ["q01"])["q01"]["segment"] == expected


# --- alignment ---------------------------------------------------------------

def test_quote_aligns_to_the_cited_segment_occurrence():
    words = _words("take it twice a day ok later again take it twice a day now", seg_every=6)
    first = align_quote(words, "twice a day")
    assert first[2] == 2  # default: the first occurrence
    cited = align_quote(words, "twice a day", segment=1)
    assert cited[2] == 10
    assert align_quote(words, "twice a day", segment=0)[2] == 2


def test_fuzzy_fallback_only_when_exact_fails():
    # The model drops the filler word the ASR kept.
    words = _words("the dose is uh 100 mg twice daily with food")
    assert align_span(words, "the dose is 100 mg twice daily") is None
    span = align_span(words, "the dose is 100 mg twice daily", fuzzy_ratio=0.8)
    assert span == (words[0]["start"], words[7]["end"])
    assert fuzzy_align(words, "completely unrelated statement here") is None


def test_offsets_shift_and_clamp():
    assert apply_offsets((1.0, 2.0), -0.2, 0.3) == (0.8, 2.3)
    assert apply_offsets((0.1, 2.0), -0.5, 0.0) == (0.0, 2.0)
    assert apply_offsets((1.0, 2.0), 0.0, 5.0, duration=2.5) == (1.0, 2.5)
    assert apply_offsets((1.0, 1.2), 0.5, -0.5) == (1.0, 1.2)  # would invert
    assert apply_offsets(None, 1.0, 1.0) is None


def test_segment_at():
    words = _words("a b c d e f", seg_every=3)
    assert segment_at(words, 1.2, 2.3) == 1


# --- the answerer ------------------------------------------------------------

TRANSCRIPT = {
    "segments": [{"id": 0, "start": 0.0, "end": 4.0, "text": "take one tablet twice a day"}],
    "words": _words("take one tablet twice a day"),
    "duration": 10.0,
}


class RecordingClient(StubClient):
    def generate(self, messages, max_new_tokens=None, max_time=None):
        self.max_time = max_time
        return super().generate(messages, max_new_tokens=max_new_tokens)


def test_truncated_reply_falls_back_per_question():
    reply = _full_reply()
    cut = reply[: reply.index('"q08"') + 3]
    answerer = LLMAnswerer(
        client=StubClient([cut]), few_shot=(), fallback=lambda qs: [False] * len(qs)
    )
    results = answerer.answer_all([f"Q{i}?" for i in range(10)], TRANSCRIPT)
    assert [r[0] for r in results] == [True] * 7 + [False] * 3
    assert all(r[1] is not None for r in results[:7])


def test_generation_error_uses_the_fallback():
    class Broken:
        def generate(self, *a, **k):
            raise RuntimeError("CUDA OOM")

    answerer = LLMAnswerer(client=Broken(), few_shot=(), fallback=lambda qs: [True, False])
    assert answerer.answer_all(["a?", "b?"], TRANSCRIPT) == [(True, None), (False, None)]


def test_deadline_bounds_generation_time():
    client = RecordingClient([_full_reply(1)])
    LLMAnswerer(client=client, few_shot=()).answer_all(
        ["Q?"], TRANSCRIPT, deadline=time.time() + 20
    )
    assert 15 < client.max_time < 19


def test_no_generation_when_the_deadline_is_too_close():
    client = RecordingClient([_full_reply(1)])
    results = LLMAnswerer(
        client=client, few_shot=(), fallback=lambda qs: [False] * len(qs)
    ).answer_all(["Q?"], TRANSCRIPT, deadline=time.time() + 1)
    assert results == [(False, None)]
    assert client.calls == []


def test_cite_segment_changes_prompt_and_alignment():
    transcript = {
        "segments": [
            {"id": 0, "start": 0.0, "end": 2.0, "text": "twice a day ok"},
            {"id": 1, "start": 2.0, "end": 4.0, "text": "again twice a day"},
        ],
        "words": _words("twice a day ok again twice a day", seg_every=4),
    }
    reply = json.dumps({"answers": [
        {"id": "q01", "answer": "yes", "segment": "s01", "evidence_quote": "twice a day"}
    ]})
    client = StubClient([reply])
    answerer = LLMAnswerer(client=client, few_shot=(), cite_segment=True)
    [(answer, span)] = answerer.answer_all(["Q?"], transcript)
    assert answer and span[0] == transcript["words"][5]["start"]
    assert '"segment"' in client.calls[0][0]["content"] + client.calls[0][-1]["content"]


def test_offsets_applied_to_spans():
    reply = json.dumps({"answers": [
        {"id": "q01", "answer": "yes", "evidence_quote": "one tablet"}
    ]})
    answerer = LLMAnswerer(client=StubClient([reply]), few_shot=(), offsets=(-0.1, 0.2))
    [(answer, span)] = answerer.answer_all(["Q?"], TRANSCRIPT)
    words = TRANSCRIPT["words"]
    assert span == (round(words[1]["start"] - 0.1, 3), round(words[2]["end"] + 0.2, 3))


def test_default_prompt_is_unchanged_without_cite_segment():
    messages = build_l1_messages(TRANSCRIPT, ["Q?"])
    assert "segment" not in messages[0]["content"]
    assert '"segment"' not in messages[-1]["content"]


# --- few-shot ----------------------------------------------------------------

def _few_shot_data():
    rows_by_tid, transcripts = {}, {}
    for n in range(6):
        tid = f"sample_{n}"
        words = _words("the dose is one hundred mg daily and more words here", seg_every=5)
        transcripts[tid] = {
            "segments": [
                {"id": 0, "start": 0.0, "end": 2.0, "text": "the dose is one hundred"},
                {"id": 1, "start": 2.0, "end": 4.0, "text": "mg daily and more words here"},
            ],
            "words": words,
        }
        rows_by_tid[tid] = [
            {"question_id": f"{tid}_p", "transcript_id": tid, "question": "Dose 100?",
             "question_type": "positive", "evidence_start": "0.0",
             "evidence_end": str(0.5 + n * 0.6)},
            {"question_id": f"{tid}_h", "transcript_id": tid, "question": "Dose 200?",
             "question_type": "hard_negative", "evidence_start": "", "evidence_end": ""},
            {"question_id": f"{tid}_o", "transcript_id": tid, "question": "Concert?",
             "question_type": "off_topic", "evidence_start": "", "evidence_end": ""},
        ]
    evidence = {f"sample_{n}_h": {"bucket": "refute"} for n in range(6)}
    return rows_by_tid, transcripts, evidence


def test_diverse_positives_span_the_duration_range():
    rows_by_tid, transcripts, evidence = _few_shot_data()
    examples = build_few_shot(rows_by_tid, transcripts, evidence, "sample_0", counts=(3, 1, 1))
    assert len(examples) == 5
    quotes = [json.loads(a)["answers"][0]["evidence_quote"] for _, a in examples[:3]]
    lengths = [len(q.split()) for q in quotes]
    assert lengths == sorted(lengths) and lengths[0] < lengths[-1]


def test_default_counts_keep_the_validated_selection():
    rows_by_tid, transcripts, evidence = _few_shot_data()
    examples = build_few_shot(rows_by_tid, transcripts, evidence, "sample_0")
    assert len(examples) == 3
    assert "Dose 100?" in examples[0][0]


def test_cite_segment_examples_carry_a_segment():
    rows_by_tid, transcripts, evidence = _few_shot_data()
    examples = build_few_shot(
        rows_by_tid, transcripts, evidence, "sample_0", cite_segment=True
    )
    positive = json.loads(examples[0][1])["answers"][0]
    negative = json.loads(examples[1][1])["answers"][0]
    assert positive["segment"] == "s00" and negative["segment"] is None


def test_strict_few_shot_raises_when_data_is_missing(monkeypatch):
    from answerers import modernbert_data

    def missing():
        raise FileNotFoundError("data/question_train.csv")

    monkeypatch.setattr(modernbert_data, "load_rows", missing)
    monkeypatch.setattr(llm_module, "_data_cache", {})
    monkeypatch.setattr(llm_module, "_few_shot_cache", {})
    with pytest.raises(FileNotFoundError):
        llm_module._build_few_shot("x", strict=True)
    assert llm_module._build_few_shot("x", strict=False) == ()


# --- prior -------------------------------------------------------------------

def test_prior_learns_off_topic_wording():
    questions = [
        "Is there any mention of attending a concert?",
        "Is there any mention of a holiday trip?",
        "Is there any mention of playing football?",
        "Was the dose 100 mg daily?",
        "Should the tablets be taken after a meal?",
        "Will the treatment last two weeks?",
    ]
    prior = QuestionPrior(epochs=200).fit(questions * 2, [0, 0, 0, 1, 1, 1] * 2)
    assert prior.predict(["Is there any mention of a concert?"]) == [False]


def test_guess_never_raises_and_matches_length():
    answers = guess(["a?", "b?", "c?"])
    assert len(answers) == 3 and all(isinstance(a, bool) for a in answers)


# --- predict: hard timeout and the GPU lock -----------------------------------

def _request(n=10):
    return ASRQuestionRequestDto(
        audio_base64=base64.b64encode(b"x").decode(),
        audio_filename="conversation_sample_17.mp3",
        questions=[f"Question {i}?" for i in range(n)],
    )


def test_hard_timeout_answers_with_guesses(monkeypatch):
    release = threading.Event()
    monkeypatch.setattr(example, "HARD_TIMEOUT_S", 0.5)
    monkeypatch.setattr(
        asr, "transcribe_bytes", lambda *a, **k: release.wait(5) or {"words": []}
    )
    started = time.time()
    response = example.predict(_request())
    assert time.time() - started < 2
    validate_response(response, 10)
    assert response.evidence_start == [None] * 10
    release.set()


def test_busy_gpu_does_not_cascade(monkeypatch):
    monkeypatch.setattr(example, "DEADLINE_S", 0.3)
    assert example._GPU_LOCK.acquire(timeout=1)
    try:
        started = time.time()
        response = example.predict(_request())
        assert time.time() - started < 2
        validate_response(response, 10)
    finally:
        example._GPU_LOCK.release()


def test_clean_span_rejects_malformed():
    assert example._clean_span((1.0, 2.0)) == (1.0, 2.0)
    assert example._clean_span((2.0, 1.0)) == (None, None)
    assert example._clean_span((float("nan"), 1.0)) == (None, None)
    assert example._clean_span(None) == (None, None)


# --- ASR hotwords ------------------------------------------------------------

def test_hotwords_keep_names_and_drop_values():
    hotwords = asr.hotwords_from_questions([
        "Should the daily dose of Pamol be 500 mg?",
        "Was esomeprazole prescribed for two weeks?",
        "Is the patient taking Ibumetin twice a day?",
    ])
    words = hotwords.split()
    assert {"Pamol", "esomeprazole", "Ibumetin", "prescribed"} <= set(words)
    assert not {"500", "mg", "two", "weeks", "twice", "day", "dose"} & set(words)


def test_hotwords_change_the_cache_key():
    assert asr.cache_path("a.mp3", b"x") != asr.cache_path("a.mp3", b"x", hotwords="Pamol")
    assert asr.cache_path("a.mp3", b"x") == asr.cache_path("a.mp3", b"x", hotwords=None)
