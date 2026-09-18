"""LLM clients for the ceiling probe.

``HFClient`` runs a local instruct model with plain ``transformers`` (one GPU,
fp16, greedy). Heavy imports are lazy so this module is import-safe in the
torch-free envs (the factory and its tests). ``StubClient`` replays canned
responses for tests and offline prompt inspection.
"""

import logging
import os
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

    def generate(self, messages: List[Dict], max_new_tokens: Optional[int] = None) -> str:
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
    ) -> None:
        self.model_name = model_name or MODEL_NAME
        self.max_new_tokens = max_new_tokens or MAX_NEW_TOKENS
        self.dtype = dtype or DTYPE
        self._model = None
        self._tokenizer = None
        self._device = "cpu"

    def _load(self) -> None:
        if self._model is not None:
            return
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        logger.info("Loading LLM %s", self.model_name)
        self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        torch_dtype = getattr(torch, self.dtype, torch.float16)
        self._model = AutoModelForCausalLM.from_pretrained(
            self.model_name, torch_dtype=torch_dtype
        )
        if DEVICE in ("cuda", "cpu"):
            self._device = DEVICE
        else:
            self._device = "cuda" if torch.cuda.is_available() else "cpu"
        self._model.to(self._device)
        self._model.eval()

    def warm_up(self) -> None:
        self._load()

    def generate(self, messages: List[Dict], max_new_tokens: Optional[int] = None) -> str:
        import torch

        self._load()
        prompt = self._tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        encoded = self._tokenizer(prompt, return_tensors="pt").to(self._device)
        with torch.no_grad():
            output = self._model.generate(
                **encoded,
                max_new_tokens=max_new_tokens or self.max_new_tokens,
                do_sample=False,
                temperature=None,
                top_p=None,
                pad_token_id=self._tokenizer.eos_token_id,
            )
        new_tokens = output[0][encoded["input_ids"].shape[1]:]
        return self._tokenizer.decode(new_tokens, skip_special_tokens=True)
