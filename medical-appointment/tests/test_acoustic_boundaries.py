"""CTC normalization and sample clocks preserve values, occurrences, and bounds."""

import copy
import json
from types import SimpleNamespace

import pytest

from answerers import acoustic_boundaries as acoustic
from answerers.acoustic_boundaries import (
    AcousticSkip, FrameClock, NormalizationError, RECIPE, boundary_proposals,
    choose_crop, convolution_geometry, encode_words, endpoint_frames, validate_token_spans,
    temporary_augmentation,
)


VOCAB = {
    "<pad>": 0, "<s>": 1, "</s>": 2, "<unk>": 3, "|": 4,
    **{chr(65 + index): 5 + index for index in range(26)}, "'": 31,
}
NUMBERS = {
    "100": "one hundred", "1000": "one thousand", "0.5": "zero point five",
    "-0.5": "minus zero point five", "+0.5": "zero point five",
    "-2": "minus two", "1": "one", "95": "ninety-five",
}


def words(tokens):
    return [
        {"word": " " + text, "start": index * 0.2, "end": index * 0.2 + 0.15}
        for index, text in enumerate(tokens)
    ]


def test_expanded_numbers_keep_original_word_owners_and_negation():
    source = words(["Do", "not", "take", "100", "mg."])
    original = copy.deepcopy(source)
    encoded = encode_words(source, VOCAB, NUMBERS.__getitem__, word_offset=20)
    assert source == original
    assert [entry["spoken"] for entry in encoded["normalizations"]] == [
        "DO", "NOT", "TAKE", "ONE HUNDRED", "MILLIGRAMS",
    ]
    assert encoded["owners"].count(23) == len("ONEHUNDRED")
    assert encoded["owners"].count(24) == len("MILLIGRAMS")
    assert {owner for owner in encoded["owners"] if owner is not None} == set(range(20, 25))
    assert len(encoded["targets"]) == len(encoded["owners"])
    assert all(token > 0 for token in encoded["targets"])


@pytest.mark.parametrize("text,expected", [
    (".5", "ZERO POINT FIVE"), ("-.5", "MINUS ZERO POINT FIVE"),
    ("+.5", "PLUS ZERO POINT FIVE"), ("\u22120.5", "MINUS ZERO POINT FIVE"),
    ("1,000,", "ONE THOUSAND"), ("95%.", "NINETY FIVE PERCENT"),
])
def test_numeric_punctuation_cannot_silently_change_the_quantity(text, expected):
    encoded = encode_words(words([text]), VOCAB, NUMBERS.__getitem__)
    assert encoded["normalizations"][0]["spoken"] == expected


def test_unit_expansion_requires_a_numeric_antecedent():
    encoded = encode_words(words(["1", "mg", "test", "mg"]), VOCAB, NUMBERS.__getitem__)
    assert [entry["spoken"] for entry in encoded["normalizations"]] == ["ONE", "MILLIGRAM", "TEST", "MG"]


def test_contractions_and_punctuation_preserve_positions_without_fake_words():
    encoded = encode_words(words(["...", "can't", ",", "stop."]), VOCAB, NUMBERS.__getitem__)
    assert len(encoded["normalizations"]) == 4
    assert encoded["normalizations"][0]["rule"] == "punctuation_only"
    assert encoded["normalizations"][1]["spoken"] == "CAN'T"
    assert {owner for owner in encoded["owners"] if owner is not None} == {1, 3}
    assert VOCAB["'"] in encoded["targets"]


def test_split_compounds_keep_their_owners_without_dropping_numeric_minus_signs():
    source = words(["follow", "-up.", "anti", "-inflammatory", "long-", "term", "-2"])
    encoded = encode_words(source, VOCAB, NUMBERS.__getitem__)
    assert [entry["spoken"] for entry in encoded["normalizations"]] == [
        "FOLLOW", "UP", "ANTI", "INFLAMMATORY", "LONG", "TERM", "MINUS TWO",
    ]
    assert encoded["normalizations"][1]["rule"] == "compound_fragment"
    assert encoded["normalizations"][3]["word_index"] == 3
    assert encoded["normalizations"][-1]["rule"] == "number"
    assert {owner for owner in encoded["owners"] if owner is not None} == set(range(7))
    with pytest.raises(NormalizationError):
        encode_words(words(["-", "2"]), VOCAB, NUMBERS.__getitem__)


@pytest.mark.parametrize("text", ["HbA1c", "120/80", "1,5", "100mg", "one/two", "r\u00e9gl\u00e9", "1000000000"])
def test_unsupported_spoken_forms_are_explicit_skips_not_deleted(text):
    with pytest.raises(NormalizationError) as caught:
        encode_words(words([text]), VOCAB, NUMBERS.__getitem__, word_offset=7)
    assert caught.value.word_index == 7


def test_missing_vocabulary_symbols_and_invalid_offsets_are_rejected():
    with pytest.raises(NormalizationError):
        encode_words(words(["Take"]), {key: value for key, value in VOCAB.items() if key != "A"}, NUMBERS.__getitem__)
    with pytest.raises(ValueError, match="offset"):
        encode_words(words(["Take"]), VOCAB, NUMBERS.__getitem__, word_offset=True)
    with pytest.raises(ValueError, match="vocabulary"):
        encode_words(words(["Take"]), {**VOCAB, "|": 0}, NUMBERS.__getitem__)
    with pytest.raises(AcousticSkip, match="no alignable"):
        encode_words(words(["..."]), VOCAB, NUMBERS.__getitem__)


def test_clock_matches_each_convolution_layer_without_duration_stretching():
    stride, receptive = convolution_geometry(RECIPE["conv_kernel"], RECIPE["conv_stride"])
    assert (stride, receptive) == (320, 400)
    for samples in (400, 401, 640, 720, 800, 16000, 16001, 480000):
        length = samples
        for kernel, step in zip(RECIPE["conv_kernel"], RECIPE["conv_stride"]):
            length = (length - kernel) // step + 1
        assert FrameClock(0, samples).frames == length
    clock = FrameClock(16000, 16000)
    assert clock.frames == 49
    assert clock.boundary(0) == 1.0
    assert clock.boundary(2) == 1.04
    assert clock.boundary(49) == 1.98
    assert clock.boundary(49) != 2.0


@pytest.mark.parametrize("origin,count", [(-320, 16000), (1, 16000), (0, 0), (True, 16000)])
def test_invalid_clock_origins_and_lengths_fail(origin, count):
    with pytest.raises(ValueError, match="sample clock"):
        FrameClock(origin, count)


def test_crop_stays_near_the_exact_word_occurrence_on_the_global_grid():
    source = [{"word": " same", "start": 30 + index * 0.2, "end": 30.15 + index * 0.2} for index in range(60)]
    first, last, clock = choose_crop(source, [35, 36], 100 * 16000)
    assert (first, last) == (27, 44)
    assert clock.origin_samples % 320 == 0
    assert clock.boundary(0) == pytest.approx(35.4)
    decorated = [{**word, "gold": [0, 1], "label": 0} for word in source]
    assert choose_crop(decorated, [35, 36], 100 * 16000) == (first, last, clock)
    with pytest.raises(ValueError, match="occurrence"):
        choose_crop(source, [True, 36], 100 * 16000)


def test_long_context_is_explicitly_ineligible_not_truncated():
    source = [{"word": " same", "start": index * 4.0, "end": index * 4.0 + 0.5} for index in range(30)]
    with pytest.raises(AcousticSkip, match="crop budget"):
        choose_crop(source, [15, 15], 120 * 16000)


def alignment_fixture():
    encoded = {"targets": [VOCAB["A"], VOCAB["|"], VOCAB["A"]], "owners": [10, None, 11]}
    spans = [
        {"token": VOCAB["A"], "start": 1, "end": 3, "log_score": -1.0},
        {"token": VOCAB["|"], "start": 4, "end": 5, "log_score": -0.5},
        {"token": VOCAB["A"], "start": 6, "end": 8, "log_score": -0.7},
    ]
    clock = FrameClock(16000, 4000)
    probabilities = [[-4.0] * len(VOCAB) for _ in range(clock.frames)]
    probabilities[6][VOCAB["A"]] = -0.01
    return encoded, spans, clock, probabilities


def test_token_owners_choose_the_right_repetition_and_retain_calibrated_baseline_once():
    encoded, spans, clock, profile = alignment_fixture()
    baseline = [1.1, 1.3]
    result = boundary_proposals(encoded, spans, [11, 11], clock, profile, baseline, 2.0)
    assert result["span"] == [1.12, 1.16]
    assert result["candidate_spans"][0] == baseline
    assert result["span"] in result["candidate_spans"]
    assert len(result["candidate_spans"]) <= 65
    assert len({tuple(span) for span in result["candidate_spans"]}) == len(result["candidate_spans"])
    assert result["start_positions"] <= 8 and result["end_positions"] <= 8
    assert all(0 <= start < end <= 2 for start, end in result["candidate_spans"])
    other = boundary_proposals(encoded, spans, [10, 10], clock, profile, baseline, 2.0)
    assert other["span"] == [1.02, 1.06]


@pytest.mark.parametrize("change", ["missing", "token", "order", "frame", "score"])
def test_alignment_token_corruption_is_rejected(change):
    encoded, spans, clock, _ = alignment_fixture()
    if change == "missing":
        spans.pop()
    elif change == "token":
        spans[0]["token"] = VOCAB["B"]
    elif change == "order":
        spans[1]["start"] = 2
    elif change == "frame":
        spans[-1]["end"] = clock.frames + 1
    else:
        spans[0]["log_score"] = float("nan")
    with pytest.raises(ValueError, match="CTC"):
        validate_token_spans(spans, encoded, clock)


def test_endpoint_underflow_is_stable_bounded_and_keeps_primary_position():
    result = endpoint_frames({"start": 4, "end": 5}, [-10000.0] * 10)
    assert result[0] == 4 and len(result) <= 8 and min(result) >= 0 and max(result) <= 10
    with pytest.raises(ValueError, match="finite"):
        endpoint_frames({"start": 4, "end": 5}, [float("nan")] * 10)
    with pytest.raises(ValueError, match="indices"):
        endpoint_frames({"start": 10, "end": 11}, [-1.0] * 10)


def test_normalizer_manifest_binds_helpers_and_protected_runtime(monkeypatch, tmp_path):
    protected = {name: "pinned" for name in (
        "torch", "torchaudio", "transformers", "faster-whisper", "ctranslate2", "huggingface-hub",
    )}
    versions = {**protected, **acoustic.PACKAGES}
    root = tmp_path / "deps"
    root.mkdir()
    manifest = {
        "complete": True, "serving_environment_modified": False, "packages": acoustic.PACKAGES,
        "protected_runtime": protected, "dependency_root": str(root),
    }
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(manifest))
    monkeypatch.syspath_prepend(str(root))
    monkeypatch.setattr(acoustic.importlib.metadata, "version", versions.__getitem__)
    module = SimpleNamespace(__file__=str(root / "num2words" / "__init__.py"), num2words=lambda value, lang: NUMBERS[value])
    original_import = acoustic.importlib.import_module
    monkeypatch.setattr(acoustic.importlib, "import_module", lambda name: module if name == "num2words" else original_import(name))
    assert acoustic.load_number_renderer(path)("0.5") == "zero point five"
    versions["torch"] = "changed"
    with pytest.raises(ValueError, match="runtime changed"):
        acoustic.load_number_renderer(path)


def test_evaluation_configuration_comparison_restores_flags_even_after_failure():
    config = SimpleNamespace(**acoustic.INFERENCE_AUGMENTATION, hidden_size=768)
    original = {"apply_spec_augment": True, "mask_time_prob": 0.05, "mask_feature_prob": 0.0}
    with pytest.raises(RuntimeError, match="forced"):
        with temporary_augmentation(config, original):
            assert config.apply_spec_augment is True and config.mask_time_prob == 0.05
            raise RuntimeError("forced comparison failure")
    assert {name: getattr(config, name) for name in acoustic.INFERENCE_AUGMENTATION} == acoustic.INFERENCE_AUGMENTATION
    assert config.hidden_size == 768
    with pytest.raises(ValueError, match="augmentation fields"):
        with temporary_augmentation(config, {"hidden_size": 1}):
            pass
