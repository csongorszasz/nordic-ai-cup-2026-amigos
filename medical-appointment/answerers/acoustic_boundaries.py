"""Occurrence-preserving CTC boundary proposals on an explicit audio sample clock."""

import importlib
import importlib.metadata
import json
import math
import re
import sys
import unicodedata
from dataclasses import dataclass
from contextlib import contextmanager
from decimal import Decimal
from pathlib import Path

from .passages import validate_word_clock


MODEL = "facebook/wav2vec2-base-960h"
REVISION = "22aad52d435eb6dbaf354bdad9b0da84ce7d6156"
PACKAGES = {"num2words": "0.5.14", "docopt": "0.6.2"}
INFERENCE_AUGMENTATION = {"apply_spec_augment": False, "mask_time_prob": 0.0, "mask_feature_prob": 0.0}
RECIPE = {
    "version": 2, "model": MODEL, "revision": REVISION,
    "sampling_rate": 16000, "conv_kernel": [10, 3, 3, 3, 3, 2, 2],
    "conv_stride": [5, 2, 2, 2, 2, 2, 2],
    "context_words_each_side": 8, "max_crop_s": 30.0,
    "endpoint_radius_frames": 4, "endpoint_quantiles": [0.1, 0.25, 0.5, 0.75, 0.9],
    "max_endpoint_positions": 8, "dtype": "float32", "device": "cpu",
    "primary_policy": "ctc_frame_cells", "extra_offset_s": [0.0, 0.0],
    "max_abs_number": "1000000000",
    "inference_augmentation": INFERENCE_AUGMENTATION,
    "eval_augmentation_equivalence_atol": 1e-5,
    "alphabetic_compound_fragments": True,
}
_NUMBER = re.compile(r"[+-]?(?:(?:\d+|\d{1,3}(?:,\d{3})+)(?:\.\d+)?|\.\d+)")
_UNITS = {
    "mg": ("milligram", "milligrams"), "mcg": ("microgram", "micrograms"),
    "\u03bcg": ("microgram", "micrograms"), "g": ("gram", "grams"),
    "kg": ("kilogram", "kilograms"), "ml": ("milliliter", "milliliters"),
    "mmol/l": ("millimole per liter", "millimoles per liter"),
}


class AcousticSkip(ValueError):
    """An explicitly ineligible source keeps the incumbent and remains scored."""


class NormalizationError(AcousticSkip):
    def __init__(self, index, text):
        self.word_index = index
        super().__init__(f"Unsupported CTC spoken form at word {index}: {text!r}")


@contextmanager
def temporary_augmentation(config, values):
    if set(values) != set(INFERENCE_AUGMENTATION):
        raise ValueError("Only the declared training-augmentation fields may be changed.")
    previous = {name: getattr(config, name) for name in values}
    try:
        for name, value in values.items():
            setattr(config, name, value)
        yield
    finally:
        for name, value in previous.items():
            setattr(config, name, value)


def load_number_renderer(manifest_path):
    manifest = json.loads(Path(manifest_path).read_text())
    if (
        manifest.get("complete") is not True or manifest.get("serving_environment_modified") is not False
        or manifest.get("packages") != PACKAGES
        or not {"torch", "torchaudio", "transformers", "faster-whisper", "ctranslate2", "huggingface-hub"}.issubset(
            manifest.get("protected_runtime", {})
        )
    ):
        raise ValueError("A complete isolated spoken-number dependency manifest is required.")
    for name, version in manifest["protected_runtime"].items():
        if importlib.metadata.version(name) != version:
            raise ValueError(f"The protected acoustic runtime changed: {name}")
    root = Path(manifest["dependency_root"]).resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"Acoustic dependency target is missing: {root}")
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    module = importlib.import_module("num2words")
    if (
        not Path(module.__file__).resolve().is_relative_to(root)
        or any(importlib.metadata.version(name) != version for name, version in PACKAGES.items())
    ):
        raise ValueError("Spoken-number helpers did not load from their pinned isolated target.")
    return lambda value: module.num2words(value, lang="en")


def encode_words(words, vocabulary, number_renderer, *, word_offset=0):
    reserved = {vocabulary.get(name) for name in ("<pad>", "<unk>", "<s>", "</s>")}
    delimiter = vocabulary.get("|")
    if (
        type(vocabulary.get("<pad>")) is not int or vocabulary["<pad>"] != 0
        or type(delimiter) is not int or delimiter < 0 or delimiter in reserved
    ):
        raise ValueError("The CTC vocabulary must declare blank zero and a word delimiter.")
    if isinstance(word_offset, bool) or not isinstance(word_offset, int) or word_offset < 0:
        raise ValueError("The original word offset must be a nonnegative integer.")
    targets, owners, trace = [], [], []
    previous_quantity = None
    for local_index, word in enumerate(words):
        index = word_offset + local_index
        original = word["word"]
        if not isinstance(original, str):
            raise NormalizationError(index, original)
        text = unicodedata.normalize("NFKC", original).strip(" \t\r\n;:!?()[]{}\"\u201c\u201d")
        text = text.replace("\u2019", "'").replace("\u2018", "'").replace("\u2212", "-")
        text = text.rstrip(".,")
        if not any(character.isdigit() for character in text):
            text = text.lstrip(".,")
        numeric = text[:-1] if text.endswith("%") else text
        quantity = None
        if _NUMBER.fullmatch(numeric):
            canonical = numeric.replace(",", "")
            quantity = Decimal(canonical)
            if abs(quantity) >= Decimal(RECIPE["max_abs_number"]):
                raise NormalizationError(index, original)
            if canonical.startswith("."):
                canonical = "0" + canonical
            elif canonical.startswith(("+.", "-.")):
                canonical = canonical[0] + "0" + canonical[1:]
            try:
                spoken = number_renderer(canonical)
            except (ValueError, NotImplementedError, OverflowError) as exc:
                raise NormalizationError(index, original) from exc
            if not isinstance(spoken, str):
                raise NormalizationError(index, original)
            if canonical.startswith("+"):
                spoken = "plus " + spoken
            rule = "number"
            if text.endswith("%"):
                spoken += " percent"
                quantity, rule = None, "percentage"
        elif text.casefold() in _UNITS and previous_quantity is not None:
            spoken = _UNITS[text.casefold()][int(previous_quantity != 1)]
            rule = "quantity_unit"
        elif not text:
            spoken, rule = "", "punctuation_only"
        elif re.fullmatch(r"-?[A-Za-z]+(?:['-][A-Za-z]+)*-?", text):
            spoken = text.strip("-")
            rule = "compound_fragment" if spoken != text else "letters"
        else:
            raise NormalizationError(index, original)
        previous_quantity = quantity
        spoken = " ".join(spoken.replace("-", " ").replace(",", " ").upper().split())
        if spoken and not re.fullmatch(r"[A-Z]+(?:'[A-Z]+)*(?: [A-Z]+(?:'[A-Z]+)*)*", spoken):
            raise NormalizationError(index, original)
        trace.append({"word_index": index, "source": original, "spoken": spoken, "rule": rule})
        for piece in spoken.split():
            if targets:
                targets.append(vocabulary["|"])
                owners.append(None)
            for character in piece:
                token = vocabulary.get(character)
                if type(token) is not int or token < 0 or token in reserved or token == delimiter:
                    raise NormalizationError(index, original)
                targets.append(token)
                owners.append(index)
    if not targets:
        raise AcousticSkip("The CTC crop contains no alignable spoken tokens.")
    return {"targets": targets, "owners": owners, "normalizations": trace}


def convolution_geometry(kernels, strides):
    if not kernels or len(kernels) != len(strides) or any(
        isinstance(value, bool) or not isinstance(value, int) or value < 1 for value in [*kernels, *strides]
    ):
        raise ValueError("Convolution kernels and strides must be matching positive integer lists.")
    stride, receptive = 1, 1
    for kernel, step in zip(kernels, strides):
        receptive += (kernel - 1) * stride
        stride *= step
    return stride, receptive


@dataclass(frozen=True)
class FrameClock:
    origin_samples: int
    sample_count: int
    sampling_rate: int = 16000
    stride_samples: int = 320
    receptive_samples: int = 400

    def __post_init__(self):
        values = (self.origin_samples, self.sample_count, self.sampling_rate, self.stride_samples, self.receptive_samples)
        if (
            any(isinstance(value, bool) or not isinstance(value, int) for value in values)
            or self.origin_samples < 0 or min(values[1:]) <= 0
            or self.origin_samples % self.stride_samples
            or self.receptive_samples < self.stride_samples
        ):
            raise ValueError("The acoustic sample clock must be positive and stride-aligned.")

    @property
    def frames(self):
        return max(0, (self.sample_count - self.receptive_samples) // self.stride_samples + 1)

    def boundary(self, frame):
        if isinstance(frame, bool) or not isinstance(frame, int) or not 0 <= frame <= self.frames:
            raise ValueError("CTC frame boundary lies outside the real unpadded crop.")
        # Snap receptive-center cell boundaries to the model's global stride grid.
        numerator = 2 * (self.origin_samples + frame * self.stride_samples) + self.receptive_samples - self.stride_samples
        sample = ((numerator + self.stride_samples) // (2 * self.stride_samples)) * self.stride_samples
        return sample / self.sampling_rate


def choose_crop(words, anchor, total_samples, sampling_rate=16000):
    if any(
        isinstance(value, bool) or not isinstance(value, int) or value <= 0
        for value in (total_samples, sampling_rate)
    ):
        raise ValueError("The audio sample count and sampling rate must be positive integers.")
    duration = total_samples / sampling_rate
    validate_word_clock(words, duration)
    if (
        not isinstance(anchor, (list, tuple)) or len(anchor) != 2
        or any(isinstance(value, bool) or not isinstance(value, int) for value in anchor)
        or not 0 <= anchor[0] <= anchor[1] < len(words)
    ):
        raise ValueError("Conditional alignment requires the exact original word occurrence.")
    first = max(0, anchor[0] - RECIPE["context_words_each_side"])
    last = min(len(words) - 1, anchor[1] + RECIPE["context_words_each_side"])
    stride, receptive = convolution_geometry(RECIPE["conv_kernel"], RECIPE["conv_stride"])
    origin = max(0, math.floor(words[first]["start"] * sampling_rate / stride + 1e-9) * stride)
    stop = min(total_samples, math.ceil(words[last]["end"] * sampling_rate - 1e-9))
    if stop <= origin or (stop - origin) / sampling_rate > RECIPE["max_crop_s"]:
        raise AcousticSkip("The whole anchored context does not fit the acoustic crop budget.")
    clock = FrameClock(origin, stop - origin, sampling_rate, stride, receptive)
    if not clock.frames:
        raise AcousticSkip("The anchored audio crop is shorter than the receptive field.")
    return first, last, clock


def validate_token_spans(spans, encoded, clock):
    if not encoded["targets"] or len(spans) != len(encoded["targets"]) or len(encoded["owners"]) != len(spans):
        raise ValueError("CTC alignment omitted or duplicated target tokens.")
    previous_end = 0
    for expected, span in zip(encoded["targets"], spans):
        if (
            span["token"] != expected
            or any(isinstance(span[key], bool) or not isinstance(span[key], int) for key in ("start", "end"))
            or not previous_end <= span["start"] < span["end"] <= clock.frames
            or not math.isfinite(span["log_score"])
        ):
            raise ValueError("CTC token order, bounds, or score is invalid.")
        previous_end = span["end"]


def endpoint_frames(token_span, log_profile, *, end=False):
    if not log_profile or any(not math.isfinite(value) for value in log_profile):
        raise ValueError("Endpoint candidates require finite acoustic activations.")
    count = len(log_profile)
    if (
        any(isinstance(token_span[key], bool) or not isinstance(token_span[key], int) for key in ("start", "end"))
        or not 0 <= token_span["start"] < token_span["end"] <= count
    ):
        raise ValueError("Token-boundary indices lie outside the acoustic activations.")
    primary = token_span["end"] if end else token_span["start"]
    low = max(0, token_span["start"] - RECIPE["endpoint_radius_frames"])
    high = min(count, token_span["end"] + RECIPE["endpoint_radius_frames"])
    selected = [primary]
    values = log_profile[low:high]
    peak = max(values)
    weights = [math.exp(value - peak) for value in values]
    total = math.fsum(weights)
    for quantile in RECIPE["endpoint_quantiles"]:
        cumulative = 0.0
        for offset, weight in enumerate(weights):
            cumulative += weight
            if cumulative >= quantile * total:
                selected.append(low + offset + int(end))
                break
    selected.extend((primary - 1, primary + 1))
    return list(dict.fromkeys(value for value in selected if 0 <= value <= count))[:RECIPE["max_endpoint_positions"]]


def boundary_proposals(encoded, spans, anchor, clock, log_probs, baseline_span, duration):
    validate_token_spans(spans, encoded, clock)
    if (
        len(log_probs) != clock.frames
        or any(len(row) <= max(encoded["targets"]) for row in log_probs)
        or not isinstance(baseline_span, (list, tuple)) or len(baseline_span) != 2
        or any(isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value)
               for value in (*baseline_span, duration))
        or not 0 <= baseline_span[0] < baseline_span[1] <= duration
    ):
        raise ValueError("Acoustic activations or retained baseline bounds are invalid.")
    relevant = [
        span for owner, span in zip(encoded["owners"], spans)
        if owner is not None and anchor[0] <= owner <= anchor[1]
    ]
    if not relevant:
        raise AcousticSkip("The selected source occurrence contains no aligned spoken tokens.")
    first, last = relevant[0], relevant[-1]
    primary = [clock.boundary(first["start"]), clock.boundary(last["end"])]
    if not 0 <= primary[0] < primary[1] <= duration:
        raise ValueError("The conditional CTC span violates the real audio bounds.")
    starts = endpoint_frames(first, [row[first["token"]] for row in log_probs])
    ends = endpoint_frames(last, [row[last["token"]] for row in log_probs], end=True)
    candidates = [list(baseline_span)]
    seen = {tuple(baseline_span)}
    for start in starts:
        for end in ends:
            span = [clock.boundary(start), clock.boundary(end)]
            if 0 <= span[0] < span[1] <= duration and tuple(span) not in seen:
                candidates.append(span)
                seen.add(tuple(span))
    return {
        "span": primary, "candidate_spans": candidates,
        "start_positions": len(starts), "end_positions": len(ends),
        "mean_selected_token_logp": math.fsum(span["log_score"] for span in relevant) / len(relevant),
    }


class ConditionalCTC:
    def __init__(self, number_renderer):
        self.number_renderer = number_renderer
        self.processor = None
        self.model = None
        self.runtime = {}
        self.original_augmentation = None

    def load(self):
        if self.model is not None:
            return
        import os
        import torch
        import torchaudio
        from transformers import AutoConfig, AutoModelForCTC, AutoProcessor

        if torch.cuda.is_available():
            raise RuntimeError("The conditional CTC probe requires a CPU-only allocation.")
        threads = int(os.environ.get("SLURM_CPUS_PER_TASK", "1"))
        if threads < 1:
            raise ValueError("Allocated CPU thread count must be positive.")
        torch.set_num_threads(threads)
        torch.set_num_interop_threads(1)
        if not callable(getattr(torchaudio.functional, "forced_align", None)):
            raise RuntimeError("The isolated runtime does not expose CTC forced alignment.")
        processor = AutoProcessor.from_pretrained(
            MODEL, revision=REVISION, local_files_only=True, trust_remote_code=False,
        )
        config = AutoConfig.from_pretrained(
            MODEL, revision=REVISION, local_files_only=True, trust_remote_code=False,
        )
        original_augmentation = {name: getattr(config, name) for name in INFERENCE_AUGMENTATION}
        for name, value in INFERENCE_AUGMENTATION.items():
            setattr(config, name, value)
        model, loading = AutoModelForCTC.from_pretrained(
            MODEL, revision=REVISION, config=config, local_files_only=True, trust_remote_code=False,
            dtype=torch.float32, attn_implementation="eager", output_loading_info=True,
        )
        model.eval()
        if loading["missing_keys"] or loading["unexpected_keys"] or loading.get("mismatched_keys") or loading.get("error_msgs"):
            raise ValueError(f"The pinned CTC weights did not load exactly: {loading}")
        if (
            list(model.config.conv_kernel) != RECIPE["conv_kernel"]
            or list(model.config.conv_stride) != RECIPE["conv_stride"]
            or processor.feature_extractor.sampling_rate != RECIPE["sampling_rate"]
            or model.config.pad_token_id != 0
            or next(model.parameters()).dtype != torch.float32
        ):
            raise ValueError("The loaded acoustic model changed its clock, vocabulary, or precision.")
        self.runtime = {
            name: importlib.metadata.version(name)
            for name in ("torch", "torchaudio", "transformers", "huggingface-hub", "num2words")
        }
        self.runtime["threads"] = torch.get_num_threads()
        self.runtime["checkpoint_augmentation"] = original_augmentation
        self.runtime["eval_augmentation_max_abs_delta"] = None
        self.original_augmentation = original_augmentation
        self.processor, self.model = processor, model

    def align(self, waveform, words, anchor, baseline_span):
        self.load()
        import torch
        import torchaudio.functional as functional

        first, last, clock = choose_crop(words, anchor, len(waveform))
        encoded = encode_words(
            words[first:last + 1], self.processor.tokenizer.get_vocab(),
            self.number_renderer, word_offset=first,
        )
        repeats = sum(left == right for left, right in zip(encoded["targets"], encoded["targets"][1:]))
        if clock.frames < len(encoded["targets"]) + repeats:
            raise AcousticSkip("The real crop has too few frames for the complete CTC target.")
        cropped = waveform[clock.origin_samples:clock.origin_samples + clock.sample_count]
        inputs = self.processor(
            cropped, sampling_rate=clock.sampling_rate, return_tensors="pt", padding=False,
        )
        if inputs["input_values"].shape[-1] != clock.sample_count:
            raise ValueError("The acoustic processor changed or padded the crop's sample clock.")
        with torch.inference_mode():
            logits = self.model(**inputs).logits
            if logits.shape[1] != clock.frames or not torch.isfinite(logits).all():
                raise ValueError("Acoustic frame count or logits differ from the explicit convolution clock.")
            if self.runtime["eval_augmentation_max_abs_delta"] is None:
                with temporary_augmentation(self.model.config, self.original_augmentation):
                    reference = self.model(**inputs).logits
                delta = float((reference - logits).abs().max())
                if not math.isfinite(delta) or delta > RECIPE["eval_augmentation_equivalence_atol"]:
                    raise RuntimeError("Disabling training-only augmentation changed evaluation logits.")
                self.runtime["eval_augmentation_max_abs_delta"] = delta
            log_probs = logits.log_softmax(dim=-1)
            if not torch.isfinite(log_probs).all():
                raise ValueError("The CTC acoustic log probabilities are not finite.")
            targets = torch.tensor([encoded["targets"]], dtype=torch.int32)
            path, scores = functional.forced_align(log_probs, targets, blank=0)
            merged = functional.merge_tokens(path[0], scores[0], blank=0)
        spans = [
            {"token": int(item.token), "start": int(item.start), "end": int(item.end), "log_score": float(item.score)}
            for item in merged
        ]
        result = boundary_proposals(
            encoded, spans, anchor, clock, log_probs[0].tolist(),
            baseline_span, len(waveform) / clock.sampling_rate,
        )
        result.update({
            "source_word_range": list(anchor),
            "context_word_range": [first, last], "normalizations": encoded["normalizations"],
            "crop_origin_samples": clock.origin_samples, "crop_samples": clock.sample_count,
            "frames": clock.frames, "target_tokens": len(encoded["targets"]),
            "stride_samples": clock.stride_samples, "receptive_samples": clock.receptive_samples,
        })
        return result
