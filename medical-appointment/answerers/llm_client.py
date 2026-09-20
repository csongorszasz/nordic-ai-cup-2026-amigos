"""LLM clients for the ceiling probe.

``HFClient`` runs a local instruct model with plain ``transformers`` (one GPU,
fp16, greedy). Heavy imports are lazy so this module is import-safe in the
torch-free envs (the factory and its tests). ``StubClient`` replays canned
responses for tests and offline prompt inspection.
"""

import logging
import os
import time
from typing import Dict, List, Optional, Sequence

logger = logging.getLogger(__name__)

MODEL_NAME = os.environ.get("MEDAPP_LLM_MODEL", "Qwen/Qwen2.5-7B-Instruct")
MAX_NEW_TOKENS = int(os.environ.get("MEDAPP_LLM_MAX_NEW_TOKENS", "512"))
DEVICE = os.environ.get("MEDAPP_LLM_DEVICE", "auto").lower()
DTYPE = os.environ.get("MEDAPP_LLM_DTYPE", "float16")


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
        enable_thinking: Optional[bool] = None,
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
        if enable_thinking is not None and not isinstance(enable_thinking, bool):
            raise ValueError("enable_thinking must be a boolean or None.")
        self.enable_thinking = enable_thinking
        self.legacy_special_tokens = (
            os.environ.get("MEDAPP_LLM_LEGACY_SPECIAL_TOKENS", "0") == "1"
            if legacy_special_tokens is None else legacy_special_tokens
        )
        self._model = None
        self._tokenizer = None
        self._device = "cpu"
        self._warmed = False
        self.last_generation = None

    def chat_template_options(self):
        return {} if self.enable_thinking is None else {"enable_thinking": self.enable_thinking}

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

            self._model = _from(AutoModelForImageTextToText)
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
        self.last_generation = None
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
            **self.chat_template_options(),
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
        started = time.monotonic()
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
        self.last_generation = {
            "prompt_tokens": int(encoded["input_ids"].shape[1]),
            "completion_tokens": int(new_tokens.numel()),
            "generation_s": time.monotonic() - started,
            "hit_token_limit": int(new_tokens.numel()) >= token_limit,
        }
        return self._tokenizer.decode(new_tokens, skip_special_tokens=True)
