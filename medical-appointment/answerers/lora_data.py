"""Pure quote-localizer data and target masking, without model dependencies."""

import json

from .align import text_between
from .llm_prompt import serialize_transcript
from .modernbert_data import grouped_folds


SYSTEM = (
    "The yes/no decision has already been established as yes. Locate the "
    "reference evidence in the consultation transcript. Copy one contiguous "
    "verbatim quote supporting the queried fact, including its needed qualifiers. "
    "A quote may be a fragment because the surrounding transcript supplies context. "
    'Return only JSON: {"evidence_quote":"exact transcript words"}.'
)


def quote_messages(question, transcript):
    return [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": f"TRANSCRIPT\n{serialize_transcript(transcript)}\n\nQUESTION\n{question}"},
    ]


def split_rows(rows, excluded, fold_index=0, seed=13):
    selected = [row for row in rows if row["transcript_id"] not in excluded]
    tids = {row["transcript_id"] for row in selected}
    folds = grouped_folds(tids, 5, seed)
    if not 0 <= fold_index < len(folds):
        raise ValueError("Fold index must identify one of the five fixed groups.")
    held = set(folds[fold_index])
    training = [row for row in selected if row["transcript_id"] not in held and row["label"] == 1]
    validation = [row for row in selected if row["transcript_id"] in held]
    if {row["transcript_id"] for row in training} & held:
        raise ValueError("Training conversations overlap validation.")
    return training, validation, folds


def encode_completion(tokenizer, messages, completion, max_length=4096):
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    complete = tokenizer.apply_chat_template(
        [*messages, {"role": "assistant", "content": completion}],
        tokenize=False, add_generation_prompt=False,
    )
    prompt_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
    full_ids = tokenizer(complete, add_special_tokens=False)["input_ids"]
    if full_ids[:len(prompt_ids)] != prompt_ids or len(full_ids) <= len(prompt_ids):
        raise ValueError("Assistant completion is not an exact continuation of the inference prompt.")
    if len(full_ids) > max_length:
        raise ValueError(f"Training sequence has {len(full_ids)} tokens, exceeding {max_length}; do not truncate evidence.")
    return {
        "input_ids": full_ids,
        "attention_mask": [1] * len(full_ids),
        "labels": [-100] * len(prompt_ids) + full_ids[len(prompt_ids):],
    }


def training_example(row, transcript, tokenizer):
    quote = text_between(transcript["words"], *row["gold"])
    if not quote:
        raise ValueError(f"Reference contains no ASR words: {row['question_id']}")
    return encode_completion(
        tokenizer, quote_messages(row["question"], transcript),
        json.dumps({"evidence_quote": quote}),
    )
