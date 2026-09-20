"""Complete-attempt scoring and legal candidate-oracle coverage."""

import local_evaluator as evaluator
from answerers.passages import Passage
from answerers.rag import union_oracle_tiou


def conversations(count):
    return [
        (
            f"conversation_{index}.mp3",
            [
                {
                    "question_id": f"{index}_yes",
                    "question": "Was the dose 100 mg?",
                    "label": "1", "question_type": "positive",
                    "evidence_start": "1", "evidence_end": "2",
                },
                {
                    "question_id": f"{index}_no",
                    "question": "Was the dose 200 mg?",
                    "label": "0", "question_type": "hard_negative",
                    "evidence_start": "", "evidence_end": "",
                },
            ],
        )
        for index in range(count)
    ]


def test_abort_counts_unsent_questions(monkeypatch):
    monkeypatch.setattr(evaluator, "group_questions_by_conversation", lambda: conversations(7))
    monkeypatch.setattr(evaluator, "load_sample_audio", lambda _: b"mp3")
    monkeypatch.setattr(
        evaluator, "_ask",
        lambda *args: ([-1, -1], [None, None], None, "timeout", True),
    )
    received = []
    stats = evaluator.replay(
        "unused", verbose=False,
        on_response=lambda *args: received.append(args),
    )
    assert stats.aborted
    assert stats.total == 14
    assert stats.errors == 14
    assert stats.timeouts == 5
    assert stats.unsent_conversations == 2
    assert len(stats.tious) == 7
    assert len(received) == 5
    assert received[0][-1] == "timeout"


def test_attempt_budget_counts_every_question(monkeypatch):
    monkeypatch.setattr(evaluator, "group_questions_by_conversation", lambda: conversations(2))
    clock = iter([0.0, 121.0])
    monkeypatch.setattr(evaluator.time, "monotonic", lambda: next(clock))
    stats = evaluator.replay("unused", verbose=False)
    assert stats.total == 4
    assert stats.unsent_conversations == 2
    assert stats.final_score == 0.0


def test_candidate_oracle_cannot_bridge_uncited_regions():
    words = [
        {"word": str(index), "start": float(index), "end": float(index + 1)}
        for index in range(5)
    ]
    candidates = [
        Passage(0, 0, 0, 0.0, 1.0, "0"),
        Passage(1, 4, 4, 4.0, 5.0, "4"),
    ]
    assert union_oracle_tiou(words, candidates, (0.0, 5.0)) == 0.2
