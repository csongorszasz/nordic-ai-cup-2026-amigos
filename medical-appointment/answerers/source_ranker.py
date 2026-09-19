"""Offline inverse-question likelihood with occurrence-scoped context ablations."""

import importlib.metadata
import math
import os
import time

from windows import join_words


MODEL = "google/flan-t5-base"
REVISION = "7bcac572ce56db69c1ea7c8af255c5d7c9672fc2"
MODES = ("source_only", "with_context", "masked_source")
POLICIES = ("with_context", "source_only", "support_gain")
INSTRUCTION = (
    "Write a specific yes/no question whose yes answer is supported by the selected "
    "passage from a doctor-patient consultation. Use surrounding context only to "
    "understand the selected passage. Focus on its fact or event."
)
RANK_RECIPE = {
    "version": 1, "model": MODEL, "revision": REVISION, "dtype": "float32", "device": "cpu",
    "source_token_limit": 512, "question_token_limit": 128, "batch_size": 8,
    "modes": list(MODES), "policies": list(POLICIES), "primary_policy": "with_context",
    "instruction": INSTRUCTION, "include_target_eos": True,
    "normalization": "mean log probability over non-padding target tokens",
    "score_tie_atol": 1e-5,
    "training": False,
}


def source_prompt(unit, words, mode):
    if mode not in MODES:
        raise ValueError(f"Unknown source-likelihood mode: {mode}")
    if not 0 <= unit.context_first_word <= unit.first_word <= unit.last_word <= unit.context_last_word < len(words):
        raise ValueError("Source and interpretation-context word ranges are inconsistent.")
    source = join_words(words, unit.first_word, unit.last_word)
    if source != unit.text:
        raise ValueError("Source text changed relative to its exact word occurrence.")
    before = join_words(words, unit.context_first_word, unit.first_word - 1)
    after = join_words(words, unit.last_word + 1, unit.context_last_word)
    if mode == "source_only":
        before = after = "(not supplied)"
    elif mode == "masked_source":
        source = "(selected passage withheld)"
    return (
        f"{INSTRUCTION}\n\nEarlier context:\n{before}\n\n"
        f"Selected passage:\n{source}\n\nLater context:\n{after}\n\nQuestion:"
    )


def question_jobs(cases, words):
    prompts, prompt_ids, jobs = [], {}, []
    for case_index, (question, units) in enumerate(cases):
        if not isinstance(question, str) or not question.strip() or not units:
            raise ValueError("Likelihood scoring requires a nonempty question and candidate list.")
        for unit_index, unit in enumerate(units):
            for mode in MODES:
                prompt = source_prompt(unit, words, mode)
                if prompt not in prompt_ids:
                    prompt_ids[prompt] = len(prompts)
                    prompts.append(prompt)
                jobs.append({
                    "case_index": case_index, "unit_index": unit_index, "mode": mode,
                    "prompt_index": prompt_ids[prompt], "target": question,
                })
    if not jobs:
        raise ValueError("No source-likelihood jobs were supplied.")
    return prompts, jobs


def select_source(scores, policy):
    if policy not in POLICIES or not scores:
        raise ValueError("A known source-selection policy and complete likelihoods are required.")
    values = []
    for entry in scores:
        if set(entry) != set(MODES) or any(
            isinstance(value, bool) or not isinstance(value, (int, float))
            or not math.isfinite(value) or value > 1e-5 for value in entry.values()
        ):
            raise ValueError("Every source likelihood must be finite and include all ablations.")
        values.append(
            entry["with_context"] - entry["masked_source"] if policy == "support_gain" else entry[policy]
        )
    best = max(values)
    return next(index for index, value in enumerate(values) if best - value <= RANK_RECIPE["score_tie_atol"])


class InverseQuestionScorer:
    def __init__(self):
        self.tokenizer = None
        self.model = None
        self.last_metrics = {}
        self.runtime = {}

    def load(self):
        if self.model is not None:
            return
        import torch
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

        if torch.cuda.is_available():
            raise RuntimeError("The inverse-question feasibility probe requires a CPU-only allocation.")
        threads = int(os.environ.get("SLURM_CPUS_PER_TASK", "1"))
        if threads < 1:
            raise ValueError("Allocated CPU threads must be positive.")
        torch.set_num_threads(threads)
        torch.set_num_interop_threads(1)
        self.tokenizer = AutoTokenizer.from_pretrained(
            MODEL, revision=REVISION, use_fast=True, local_files_only=True, trust_remote_code=False,
        )
        if self.tokenizer.padding_side != "right":
            raise ValueError("The frozen likelihood implementation requires right padding.")
        self.model = AutoModelForSeq2SeqLM.from_pretrained(
            MODEL, revision=REVISION, dtype=torch.float32, attn_implementation="eager",
            local_files_only=True, trust_remote_code=False,
        ).eval()
        if not self.model.config.is_encoder_decoder or next(self.model.parameters()).dtype != torch.float32:
            raise ValueError("The pinned encoder-decoder did not load in the declared precision.")
        self.runtime = {
            "torch": torch.__version__, "threads": torch.get_num_threads(),
            "tokenizer_class": type(self.tokenizer).__name__, "model_class": type(self.model).__name__,
            **{name: importlib.metadata.version(name) for name in ("transformers", "huggingface_hub", "tokenizers")},
        }

    def score(self, cases, words, *, check_direct=False):
        self.load()
        import torch
        from transformers.modeling_outputs import BaseModelOutput

        started = time.monotonic()
        prompts, jobs = question_jobs(cases, words)
        encoded = self.tokenizer(prompts, truncation=False, padding=False, return_token_type_ids=False)
        targets = self.tokenizer(
            [question for question, _ in cases], truncation=False, padding=False, return_token_type_ids=False,
        )
        if any(len(ids) > RANK_RECIPE["source_token_limit"] for ids in encoded["input_ids"]):
            raise ValueError("A source prompt exceeds the frozen token budget; do not truncate evidence.")
        if any(not ids or len(ids) > RANK_RECIPE["question_token_limit"] for ids in targets["input_ids"]):
            raise ValueError("A supplied question exceeds the frozen target budget.")
        if any(ids[-1] != self.tokenizer.eos_token_id for ids in targets["input_ids"]):
            raise ValueError("The supplied question target is missing its real EOS token.")
        batch_size = RANK_RECIPE["batch_size"]
        hidden = []
        encoding_started = time.monotonic()
        with torch.inference_mode():
            for first in range(0, len(prompts), batch_size):
                indices = range(first, min(first + batch_size, len(prompts)))
                batch = self.tokenizer.pad(
                    [{"input_ids": encoded["input_ids"][index], "attention_mask": encoded["attention_mask"][index]}
                     for index in indices],
                    padding=True, return_tensors="pt",
                )
                states = self.model.get_encoder()(**batch).last_hidden_state
                if not torch.isfinite(states).all():
                    raise RuntimeError("The source encoder produced non-finite states.")
                for offset, index in enumerate(indices):
                    hidden.append(states[offset, :len(encoded["input_ids"][index])].clone())
        encoding_s = time.monotonic() - encoding_started
        result = [[{} for _ in units] for _, units in cases]
        direct_delta = None
        decoding_started = time.monotonic()
        with torch.inference_mode():
            for first in range(0, len(jobs), batch_size):
                chunk = jobs[first:first + batch_size]
                states = [hidden[job["prompt_index"]] for job in chunk]
                encoder_states = torch.nn.utils.rnn.pad_sequence(states, batch_first=True)
                lengths = torch.tensor([len(state) for state in states])
                attention = (torch.arange(encoder_states.shape[1])[None, :] < lengths[:, None]).long()
                target = self.tokenizer.pad(
                    [{"input_ids": targets["input_ids"][job["case_index"]],
                      "attention_mask": targets["attention_mask"][job["case_index"]]} for job in chunk],
                    padding=True, return_tensors="pt",
                )
                labels = target["input_ids"].masked_fill(target["attention_mask"] == 0, -100)
                output = self.model(
                    encoder_outputs=BaseModelOutput(last_hidden_state=encoder_states),
                    attention_mask=attention, labels=labels,
                    decoder_attention_mask=target["attention_mask"], use_cache=False,
                )
                if output.logits.shape[:2] != labels.shape or not torch.isfinite(output.logits).all():
                    raise RuntimeError("The question decoder produced invalid token logits.")
                logp = output.logits.log_softmax(dim=-1).gather(-1, labels.clamp_min(0).unsqueeze(-1)).squeeze(-1)
                mask = labels != -100
                means = (logp * mask).sum(dim=1) / mask.sum(dim=1)
                if not torch.isfinite(means).all():
                    raise RuntimeError("The supplied-question likelihood is non-finite.")
                for job, value in zip(chunk, means.tolist()):
                    result[job["case_index"]][job["unit_index"]][job["mode"]] = value
                if check_direct and first == 0:
                    job = chunk[0]
                    source = self.tokenizer(prompts[job["prompt_index"]], return_tensors="pt", return_token_type_ids=False)
                    direct_target = self.tokenizer(job["target"], return_tensors="pt", return_token_type_ids=False)
                    direct = self.model(
                        **source, labels=direct_target["input_ids"],
                        decoder_attention_mask=direct_target["attention_mask"], use_cache=False,
                    )
                    direct_delta = abs(float(means[0]) + float(direct.loss))
                    if not math.isfinite(direct_delta) or direct_delta > RANK_RECIPE["score_tie_atol"]:
                        raise RuntimeError("Cached batched likelihood disagrees with direct teacher forcing.")
        for scores in result:
            select_source(scores, "with_context")
        self.last_metrics = {
            "cases": len(cases), "likelihood_jobs": len(jobs), "unique_encoder_prompts": len(prompts),
            "encoder_s": encoding_s, "decoder_s": time.monotonic() - decoding_started,
            "total_s": time.monotonic() - started,
            "source_tokens_max": max(map(len, encoded["input_ids"])),
            "target_tokens_max": max(map(len, targets["input_ids"])),
            "cached_direct_max_abs_delta": direct_delta,
        }
        return result
