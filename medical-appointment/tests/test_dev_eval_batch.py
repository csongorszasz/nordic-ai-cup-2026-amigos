"""dev_eval's batch-answerer branch (the ModernBERT scoring path)."""

import json
from pathlib import Path

import dev_eval

FIXTURE = Path(__file__).parent / "fixtures" / "transcript_sample.json"


def fake_conversation():
    rows = [
        {
            "question_id": "q1", "transcript_id": "sample_x",
            "question": "Was the dose 100 mg?", "label": "1",
            "question_type": "positive",
            "evidence_start": "1.0", "evidence_end": "2.0",
        },
        {
            "question_id": "q2", "transcript_id": "sample_x",
            "question": "Was the dose 200 mg?", "label": "0",
            "question_type": "hard_negative",
            "evidence_start": "", "evidence_end": "",
        },
        {
            "question_id": "q3", "transcript_id": "sample_x",
            "question": "Did they discuss migraines?", "label": "0",
            "question_type": "off_topic",
            "evidence_start": "", "evidence_end": "",
        },
    ]
    return [("conversation_sample_x.mp3", rows)]


class FakeBatch:
    def __init__(self, count=None, supports_info=True):
        self.name = "fake"
        self.count = count
        self.supports_info = supports_info

    def warm_up(self):
        pass

    def answer_all(self, questions, transcript, deadline=None, return_info=False):
        n = len(questions) if self.count is None else self.count
        out = []
        for _ in range(min(n, len(questions))):
            if return_info:
                out.append(
                    (
                        True,
                        (1.0, 2.0),
                        {"p": 0.9, "span": (0.8, 2.2)},
                    )
                )
            else:
                out.append((True, (1.0, 2.0)))
        return out


def _patch(monkeypatch):
    monkeypatch.setattr(
        dev_eval, "group_questions_by_conversation", fake_conversation
    )
    monkeypatch.setattr(dev_eval, "load_sample_audio", lambda name: b"audio")
    monkeypatch.setattr(
        dev_eval.asr, "transcribe_bytes",
        lambda *args, **kwargs: json.loads(FIXTURE.read_text()),
    )


def test_batch_branch_uses_batch_answerer(monkeypatch, tmp_path):
    _patch(monkeypatch)
    oof = tmp_path / "oof.json"

    stats = dev_eval.run(
        dev_eval.answer_all_false,
        limit=1,
        batch_answerer=FakeBatch(),
        oof_path=str(oof),
    )

    # The fake says yes to all three; only the positive label is yes.
    assert stats.total == 3
    assert stats.correct == 1
    records = json.loads(oof.read_text())
    assert len(records) == 3
    assert records[0]["p"] == 0.9
    assert records[0]["span"] == [1.0, 2.0]
    assert records[0]["proposed_span"] == [0.8, 2.2]


def test_batch_short_return_is_padded_with_guesses(monkeypatch):
    _patch(monkeypatch)

    stats = dev_eval.run(
        dev_eval.answer_all_false, limit=1, batch_answerer=FakeBatch(count=2)
    )

    assert stats.total == 3
    assert stats.correct == 2


def test_batch_without_info_support(monkeypatch):
    _patch(monkeypatch)

    stats = dev_eval.run(
        dev_eval.answer_all_false,
        limit=1,
        batch_answerer=FakeBatch(supports_info=False),
    )

    assert stats.total == 3
