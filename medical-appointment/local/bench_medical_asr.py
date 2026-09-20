"""WER benchmark: medical Whisper vs baseline Whisper on a PriMock subset.

    python local/bench_medical_asr.py --n 40 \
      --models large-v3-turbo models/medical-whisper-large-v3-ct2

References come from the dataset (`Na0s/Primock_med` test split, which pairs
16 kHz `audio` with a `sentence` transcription). Transcribes on CUDA with int8.
"""

import argparse
import re
import time

import jiwer


def normalise(text: str) -> str:
    text = re.sub(r"\s+", " ", text.lower())
    text = re.sub(r"[^a-z0-9 ]", "", text)
    return text.strip()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="Na0s/Primock_med")
    parser.add_argument("--split", default="test")
    parser.add_argument("--n", type=int, default=40)
    parser.add_argument("--models", nargs="+",
                        default=["large-v3-turbo", "models/medical-whisper-large-v3-ct2"])
    args = parser.parse_args()

    import datasets
    from faster_whisper import WhisperModel

    ds = datasets.load_dataset(args.dataset, split=args.split, streaming=True)
    samples = []
    for i, row in enumerate(ds):
        if i >= args.n:
            break
        samples.append((row["audio"]["array"], row["sentence"]))
    print(f"loaded {len(samples)} utterances from {args.dataset}:{args.split}")

    for model_name in args.models:
        try:
            model = WhisperModel(model_name, device="cuda", compute_type="int8")
        except Exception as exc:
            print(f"  {model_name}: LOAD FAILED {exc}")
            continue
        refs, hyps = [], []
        began = time.perf_counter()
        for audio, reference in samples:
            segments, _ = model.transcribe(
                audio, language="en", condition_on_previous_text=False,
                vad_filter=False,
            )
            hyps.append(" ".join(s.text for s in segments).strip())
            refs.append(reference)
        elapsed = time.perf_counter() - began
        error = jiwer.wer(
            [normalise(r) for r in refs],
            [normalise(h) for h in hyps],
        )
        print(f"  {model_name:<45} WER {error:.3f}   ({elapsed:.0f}s for {len(samples)})")
        del model
        import gc
        import torch
        gc.collect()
        torch.cuda.empty_cache()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
