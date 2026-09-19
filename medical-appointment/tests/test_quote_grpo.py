"""The reward never consults held-out examples or changes the frozen split."""

import copy
import json

import pytest

from train_quote_grpo import (
    GroundingReward, MODEL, REVISION, configure_completion_eos, prompt_record, validate_warm_start,
)


TRAINING = [{"question_id": "train_q", "transcript_id": "train", "label": 1,
             "question": "Was it near?", "gold": [1.0, 2.0]}]
VALIDATION = [{"question_id": "held_q", "transcript_id": "held", "label": 0}]
SPLIT = {
    "fold": 0, "seed": 13, "training_question_ids": ["train_q"], "training_tids": ["train"],
    "validation_tids": ["held"], "excluded_demonstration_tids": ["demo"],
}
METADATA = {"model": MODEL, "revision": REVISION, "smoke_only": False, "finite_training": True}
CONFIG = {"base_model_name_or_path": MODEL, "revision": REVISION}
TRANSCRIPT = {
    "duration": 10.0,
    "segments": [{"id": 0, "start": 0.8, "end": 5.0, "text": "near far"}],
    "words": [{"word": " near", "start": 0.8, "end": 2.0},
              {"word": " far", "start": 3.8, "end": 5.0}],
}


def test_exact_sft_split_can_be_reused():
    validate_warm_start(TRAINING, VALIDATION, {"demo"}, SPLIT, METADATA, CONFIG, 0, 13)


@pytest.mark.parametrize("key,value", [
    ("training_question_ids", ["train_q", "held_q"]),
    ("training_question_ids", ["train_q", "train_q"]),
    ("training_tids", ["train", "held"]),
    ("validation_tids", ["another"]),
    ("excluded_demonstration_tids", []),
    ("fold", 1), ("seed", 37),
])
def test_warm_start_rejects_leakage_or_different_split(key, value):
    with pytest.raises(ValueError, match="warm-start"):
        validate_warm_start(TRAINING, VALIDATION, {"demo"}, {**SPLIT, key: value},
                            METADATA, CONFIG, 0, 13)


def test_wrong_revision_or_smoke_adapter_is_not_a_warm_start():
    for metadata in ({**METADATA, "revision": "different"}, {**METADATA, "smoke_only": True}):
        with pytest.raises(ValueError, match="provenance"):
            validate_warm_start(TRAINING, VALIDATION, {"demo"}, SPLIT, metadata, CONFIG, 0, 13)


def test_reward_group_logging_and_nonconstant_signal(tmp_path):
    journal = tmp_path / "rollouts.jsonl"
    reward = GroundingReward(TRAINING, {"train": TRANSCRIPT}, 4, journal)
    completions = [json.dumps({"evidence_quote": quote}) for quote in ("near", "far", "absent", "near")]
    assert reward(completions, ["train_q"] * 4)[0] == 2.0
    assert reward.summary() == {
        "rollouts": 4, "groups": 1, "variable_reward_groups": 1,
        "grounded_rollouts": 3, "positive_iou_rollouts": 2, "mean_tiou": 0.5,
    }
    entries = [json.loads(line) for line in journal.read_text().splitlines()]
    assert entries[2]["reward"] == -1
    assert entries[2]["span"] is None


def test_reward_rejects_held_out_ids_before_scoring():
    reward = GroundingReward(TRAINING, {"train": TRANSCRIPT}, 2, None)
    with pytest.raises(ValueError, match="non-training"):
        reward(["{}", "{}"], ["held_q", "held_q"])
    with pytest.raises(ValueError, match="complete generation"):
        reward(["{}"], ["train_q"])
    with pytest.raises(ValueError, match="mixes"):
        reward(["{}", "{}"], ["train_q", "held_q"])


def test_prompt_fields_do_not_include_gold_or_prediction():
    class Tokenizer:
        def apply_chat_template(self, messages, **kwargs):
            assert set(kwargs) == {"tokenize", "add_generation_prompt"}
            return json.dumps(messages)

        def __call__(self, text, **kwargs):
            assert kwargs == {"add_special_tokens": False}
            return {"input_ids": list(text.encode())}

    row = copy.deepcopy(TRAINING[0])
    record, ids = prompt_record(row, TRANSCRIPT, Tokenizer())
    other, _ = prompt_record({**row, "gold": [98, 99], "answer": False}, TRANSCRIPT, Tokenizer())
    assert record == other
    assert set(record) == {"prompt", "question_id"}
    assert ids and "gold" not in record["prompt"]


def test_grpo_uses_assistant_terminator_not_general_eos():
    class Tokenizer:
        eos_token = "general"

        @property
        def eos_token_id(self):
            return 1 if self.eos_token == "general" else 106

        def apply_chat_template(self, messages, tokenize, add_generation_prompt):
            return "prefix" if add_generation_prompt else "complete"

        def __call__(self, text, **kwargs):
            return {"input_ids": [2, 7] if text == "prefix" else [2, 7, 9, 106, 3]}

        def convert_ids_to_tokens(self, index):
            return "turn_end" if index == 106 else "general"

    tokenizer = Tokenizer()
    assert configure_completion_eos(tokenizer, [1, 106, 50]) == 106
    assert tokenizer.eos_token_id == 106
    with pytest.raises(ValueError, match="no native"):
        configure_completion_eos(Tokenizer(), [1])
