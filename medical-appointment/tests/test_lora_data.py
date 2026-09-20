"""No reference leakage into localizer prompts, folds, or causal loss masking."""

import pytest

from answerers.lora_data import encode_completion, quote_messages, split_rows


class Tokenizer:
    def apply_chat_template(self, messages, tokenize, add_generation_prompt):
        return "prefix" if add_generation_prompt else "prefix" + messages[-1]["content"]

    def __call__(self, text, add_special_tokens):
        assert add_special_tokens is False
        return {"input_ids": list(text.encode())}


def test_only_completion_tokens_are_supervised():
    result = encode_completion(Tokenizer(), [{"role": "user", "content": "Q"}], "answer")
    assert result["labels"][:6] == [-100] * 6
    assert result["labels"][6:] == list(b"answer")
    assert len(result["input_ids"]) == len(result["labels"])
    with pytest.raises(ValueError, match="do not truncate"):
        encode_completion(Tokenizer(), [], "answer", max_length=3)


def test_held_out_and_demonstration_conversations_are_excluded():
    rows = [
        {"question_id": f"{tid}_{label}", "transcript_id": f"s{tid}", "label": label}
        for tid in range(11) for label in (0, 1)
    ]
    training, validation, folds = split_rows(rows, {"s0"}, 0, 13)
    held = set(folds[0])
    assert all(row["label"] == 1 for row in training)
    assert {row["transcript_id"] for row in training}.isdisjoint(held | {"s0"})
    assert {row["transcript_id"] for row in validation} == held


def test_prompt_contains_no_reference_fields():
    transcript = {"segments": [{"id": 0, "start": 0.0, "end": 1.0, "text": "The dose is 100 mg."}]}
    messages = quote_messages("Was the dose 100 mg?", transcript)
    assert "100 mg" in messages[-1]["content"]
    assert "gold" not in messages[-1]["content"]
    assert "question_type" not in messages[-1]["content"]
