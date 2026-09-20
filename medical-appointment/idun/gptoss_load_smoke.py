"""Minimal gpt-oss-120b load/generate smoke. Development-only, no serving."""

import os
import time

# Allow the `kernels` package to fetch MXFP4 kernel binaries; the model itself
# is loaded with local_files_only=True so no weights are re-downloaded.
os.environ["HF_HUB_OFFLINE"] = "0"

import torch  # noqa: E402
from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: E402

MODEL = "openai/gpt-oss-120b"
REVISION = "b5c939de8f754692c1647ca79fbf85e8c1e70f8a"


def main() -> int:
    print("torch", torch.__version__, "cuda", torch.cuda.is_available())
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL, revision=REVISION, local_files_only=True
    )
    started = time.time()
    try:
        model = AutoModelForCausalLM.from_pretrained(
            MODEL, revision=REVISION, local_files_only=True, dtype=torch.bfloat16
        )
    except TypeError as exc:
        print("dtype kwarg failed:", exc)
        model = AutoModelForCausalLM.from_pretrained(
            MODEL, revision=REVISION, local_files_only=True
        )
    print(f"loaded in {time.time() - started:.1f}s")
    model = model.to("cuda").eval()
    print(
        "cuda mem allocated %.1f GB, max %.1f GB"
        % (torch.cuda.memory_allocated() / 1e9, torch.cuda.max_memory_allocated() / 1e9)
    )

    messages = [{"role": "user", "content": "Reply with exactly the word: OK"}]
    text = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
    inputs = tokenizer(text, return_tensors="pt").to("cuda")
    started = time.time()
    output = model.generate(**inputs, max_new_tokens=16, do_sample=False)
    print(f"generated in {time.time() - started:.1f}s")
    print(
        "OUT:",
        repr(tokenizer.decode(output[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
