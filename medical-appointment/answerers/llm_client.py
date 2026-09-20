"""LLM clients for the ceiling probe.

``HFClient`` runs a local instruct model with plain ``transformers`` (one GPU,
fp16, greedy). Heavy imports are lazy so this module is import-safe in the
torch-free envs (the factory and its tests). ``StubClient`` replays canned
responses for tests and offline prompt inspection.
"""

import logging
import os
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence

logger = logging.getLogger(__name__)

MODEL_NAME = os.environ.get("MEDAPP_LLM_MODEL", "Qwen/Qwen2.5-7B-Instruct")
MAX_NEW_TOKENS = int(os.environ.get("MEDAPP_LLM_MAX_NEW_TOKENS", "512"))
DEVICE = os.environ.get("MEDAPP_LLM_DEVICE", "auto").lower()
DTYPE = os.environ.get("MEDAPP_LLM_DTYPE", "float16")


def chat_template_kwargs() -> Dict:
    """Extra chat-template variables, empty unless explicitly requested.

    Thinking models (e.g. Qwen3.8) emit a reasoning block by default, which
    would hide the strict JSON citation from the parser. Set
    ``MEDAPP_LLM_ENABLE_THINKING=0`` to ask for a direct answer;
    ``MEDAPP_LLM_REASONING_EFFORT`` tunes the depth. Both are unset by default
    so non-thinking checkpoints and their templates are unchanged.
    """
    kwargs: Dict = {}
    thinking = os.environ.get("MEDAPP_LLM_ENABLE_THINKING")
    if thinking is not None:
        kwargs["enable_thinking"] = thinking != "0"
    effort = os.environ.get("MEDAPP_LLM_REASONING_EFFORT")
    if effort:
        kwargs["reasoning_effort"] = effort
    return kwargs


def token_logprobs(logits, token_ids) -> List[float]:
    """Per-step log-probability of ``token_ids`` under ``logits`` (numpy, pure)."""
    import numpy as np

    logits = np.asarray(logits, dtype=np.float64)
    ids = np.asarray(token_ids, dtype=int)
    if logits.ndim != 2 or ids.ndim != 1 or logits.shape[0] != ids.shape[0]:
        raise ValueError("logits must be [T, V] and token_ids must be [T].")
    maximum = logits.max(axis=1, keepdims=True)
    log_sum_exp = maximum[:, 0] + np.log(np.exp(logits - maximum).sum(axis=1))
    return (logits[np.arange(ids.shape[0]), ids] - log_sum_exp).tolist()


def mean_token_logprob(logits, token_ids) -> float:
    """Mean per-token log-probability of a completion (higher = more confident)."""
    values = token_logprobs(logits, token_ids)
    return float(sum(values) / len(values)) if values else 0.0


class StubClient:
    """Deterministic client for tests and dry prompt rendering."""

    def __init__(self, responses: Sequence[str]):
        self.responses = list(responses)
        self.calls: List[List[Dict]] = []

    def generate(
        self, messages: List[Dict], max_new_tokens: Optional[int] = None,
        *, deadline: Optional[float] = None,
    ) -> str:
        self.calls.append(messages)
        if not self.responses:
            return ""
        if len(self.responses) == 1:
            return self.responses[0]
        return self.responses.pop(0)


class HFClient:
    """Local HuggingFace causal LM, greedy decoding, one GPU."""

    def __init__(
        self,
        model_name: Optional[str] = None,
        max_new_tokens: Optional[int] = None,
        dtype: Optional[str] = None,
        revision: Optional[str] = None,
        legacy_special_tokens: Optional[bool] = None,
        num_beams: Optional[int] = None,
    ) -> None:
        self.model_name = model_name or MODEL_NAME
        self.max_new_tokens = MAX_NEW_TOKENS if max_new_tokens is None else max_new_tokens
        if isinstance(self.max_new_tokens, bool) or not isinstance(self.max_new_tokens, int) or self.max_new_tokens < 1:
            raise ValueError("max_new_tokens must be a positive integer.")
        self.dtype = dtype or DTYPE
        self.revision = revision or os.environ.get("MEDAPP_LLM_REVISION") or None
        self.num_beams = int(os.environ.get("MEDAPP_LLM_NUM_BEAMS", "1")) if num_beams is None else num_beams
        if isinstance(self.num_beams, bool) or not isinstance(self.num_beams, int) or self.num_beams < 1:
            raise ValueError("num_beams must be a positive integer.")
        self.legacy_special_tokens = (
            os.environ.get("MEDAPP_LLM_LEGACY_SPECIAL_TOKENS", "0") == "1"
            if legacy_special_tokens is None else legacy_special_tokens
        )
        self._model = None
        self._tokenizer = None
        self._device = "cpu"
        self._warmed = False

    def _load(self) -> None:
        if self._model is not None:
            return
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        logger.info("Loading LLM %s", self.model_name)
        try:
            self._tokenizer = AutoTokenizer.from_pretrained(
                self.model_name, revision=self.revision, local_files_only=True
            )
        except (ValueError, TypeError):
            from transformers import AutoProcessor

            self._tokenizer = AutoProcessor.from_pretrained(
                self.model_name, revision=self.revision, local_files_only=True
            ).tokenizer

        template_path = os.environ.get("MEDAPP_LLM_CHAT_TEMPLATE")
        if template_path:
            self._tokenizer.chat_template = Path(template_path).read_text(encoding="utf-8")
            logger.info("Using chat template from %s", template_path)

        if self.dtype not in ("float16", "bfloat16", "float32"):
            raise ValueError(f"Unsupported LLM dtype: {self.dtype}")
        torch_dtype = getattr(torch, self.dtype)

        # transformers >=5 renamed `torch_dtype` to `dtype`; passing the old name
        # is silently ignored, which loads fp32 and OOMs a 7B on a 32 GB card.
        def _from(factory):
            try:
                return factory.from_pretrained(
                    self.model_name, dtype=torch_dtype, revision=self.revision,
                    local_files_only=True,
                )
            except TypeError:
                return factory.from_pretrained(
                    self.model_name, torch_dtype=torch_dtype, revision=self.revision,
                    local_files_only=True,
                )

        try:
            self._model = _from(AutoModelForCausalLM)
        except ValueError as exc:
            # Multimodal checkpoints (e.g. google/gemma-4-e4b-it, whose config
            # architecture is Gemma4ForConditionalGeneration) are not a
            # CausalLM; load the image-text-to-text class and generate text.
            logger.warning("CausalLM load failed (%s); trying image-text-to-text.", exc)
            from transformers import AutoModelForImageTextToText

            try:
                self._model = _from(AutoModelForImageTextToText)
            except ValueError as exc2:
                # Newer multimodal architectures (e.g. Qwen3.8-27B,
                # Qwen3_5ForConditionalGeneration) expose the catch-all loader.
                logger.warning("Image-text-to-text load failed (%s); trying multimodal.", exc2)
                try:
                    from transformers import AutoModelForMultimodalLM
                except ImportError as exc3:
                    raise exc2 from exc3
                self._model = _from(AutoModelForMultimodalLM)
        if DEVICE in ("cuda", "cpu"):
            self._device = DEVICE
        else:
            self._device = "cuda" if torch.cuda.is_available() else "cpu"
        self._model.to(self._device)
        self._model.eval()
        logger.info(
            "LLM loaded: dtype=%s device=%s mem=%.2fGB",
            next(self._model.parameters()).dtype,
            self._device,
            torch.cuda.memory_allocated() / 1e9 if self._device == "cuda" else 0.0,
        )

    def warm_up(self) -> None:
        self._load()
        if not self._warmed:
            self.generate(
                [{"role": "user", "content": 'Reply with the JSON object {"ready":true}.'}],
                max_new_tokens=16,
            )
            self._warmed = True

    def generate(
        self, messages: List[Dict], max_new_tokens: Optional[int] = None,
        *, deadline: Optional[float] = None,
    ) -> str:
        if deadline is not None and time.monotonic() >= deadline:
            raise TimeoutError("No generation budget remains.")
        token_limit = self.max_new_tokens if max_new_tokens is None else max_new_tokens
        if isinstance(token_limit, bool) or not isinstance(token_limit, int) or token_limit < 1:
            raise ValueError("max_new_tokens must be a positive integer.")
        import torch
        from transformers import StoppingCriteria, StoppingCriteriaList

        self._load()
        prompt = self._tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
            **chat_template_kwargs(),
        )
        encoded = self._tokenizer(
            prompt, return_tensors="pt", add_special_tokens=self.legacy_special_tokens
        ).to(self._device)
        logger.info("prompt tokens=%d", encoded["input_ids"].shape[1])
        criteria = StoppingCriteriaList()
        if deadline is not None:
            class Deadline(StoppingCriteria):
                def __call__(self, input_ids, scores, **kwargs):
                    return time.monotonic() >= deadline

            if time.monotonic() >= deadline:
                raise TimeoutError("No generation budget remains after tokenization.")
            criteria.append(Deadline())
        with torch.no_grad():
            output = self._model.generate(
                **encoded,
                max_new_tokens=token_limit,
                do_sample=False,
                num_beams=self.num_beams,
                temperature=None,
                top_p=None,
                pad_token_id=self._tokenizer.eos_token_id,
                stopping_criteria=criteria,
            )
        new_tokens = output[0][encoded["input_ids"].shape[1]:]
        return self._tokenizer.decode(new_tokens, skip_special_tokens=True)

    def generate_scored(
        self, messages: List[Dict], max_new_tokens: Optional[int] = None,
        *, deadline: Optional[float] = None,
    ):
        """Greedy generation plus per-token log-probabilities of the completion.

        Returns ``(text, logprobs, token_ids)`` with ``logprobs`` aligned to the
        generated tokens. Requires greedy decoding (``num_beams=1``); used only
        by the confidence diagnostic, never by ``/predict``.
        """
        if self.num_beams != 1:
            raise ValueError("generate_scored requires greedy decoding (num_beams=1).")
        if deadline is not None and time.monotonic() >= deadline:
            raise TimeoutError("No generation budget remains.")
        token_limit = self.max_new_tokens if max_new_tokens is None else max_new_tokens
        if isinstance(token_limit, bool) or not isinstance(token_limit, int) or token_limit < 1:
            raise ValueError("max_new_tokens must be a positive integer.")
        import torch
        from transformers import StoppingCriteria, StoppingCriteriaList

        self._load()
        prompt = self._tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
            **chat_template_kwargs(),
        )
        encoded = self._tokenizer(
            prompt, return_tensors="pt", add_special_tokens=self.legacy_special_tokens
        ).to(self._device)
        logger.info("prompt tokens=%d", encoded["input_ids"].shape[1])
        criteria = StoppingCriteriaList()
        if deadline is not None:
            class Deadline(StoppingCriteria):
                def __call__(self, input_ids, scores, **kwargs):
                    return time.monotonic() >= deadline

            if time.monotonic() >= deadline:
                raise TimeoutError("No generation budget remains after tokenization.")
            criteria.append(Deadline())
        with torch.no_grad():
            output = self._model.generate(
                **encoded,
                max_new_tokens=token_limit,
                do_sample=False,
                num_beams=1,
                temperature=None,
                top_p=None,
                pad_token_id=self._tokenizer.eos_token_id,
                stopping_criteria=criteria,
                output_scores=True,
                return_dict_in_generate=True,
            )
        new_tokens = output.sequences[0][encoded["input_ids"].shape[1]:]
        if not output.scores or new_tokens.numel() == 0:
            return "", [], []
        logits = torch.stack([step[0] for step in output.scores]).float().cpu().numpy()
        ids = new_tokens.cpu().numpy()
        logprobs = token_logprobs(logits, ids)
        text = self._tokenizer.decode(new_tokens, skip_special_tokens=True)
        return text, logprobs, [int(token) for token in ids]

    def score_completion(
        self, messages: List[Dict], completion: str, *, deadline: Optional[float] = None,
    ) -> float:
        """Mean token log-probability of ``completion`` given the chat prompt.

        Teacher-forced forward pass (no generation); used only by the candidate
        selection diagnostic, never by ``/predict``.
        """
        if deadline is not None and time.monotonic() >= deadline:
            raise TimeoutError("No scoring budget remains.")
        if not completion or not completion.strip():
            return float("-inf")
        import torch

        self._load()
        prompt = self._tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
            **chat_template_kwargs(),
        )
        prompt_ids = self._tokenizer(
            prompt, add_special_tokens=self.legacy_special_tokens
        )["input_ids"]
        completion_ids = self._tokenizer(completion, add_special_tokens=False)["input_ids"]
        if not completion_ids:
            return float("-inf")
        input_ids = torch.tensor([prompt_ids + completion_ids], device=self._device)
        with torch.no_grad():
            logits = self._model(input_ids=input_ids).logits[0]
        start = len(prompt_ids)
        values = [
            float(torch.log_softmax(logits[start + i - 1].float(), dim=-1)[token])
            for i, token in enumerate(completion_ids)
        ]
        return float(sum(values) / len(values))
