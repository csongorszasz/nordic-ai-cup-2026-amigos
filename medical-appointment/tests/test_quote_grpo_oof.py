"""A pilot cannot be silently retuned or mixed with another GRPO runtime."""

import copy
import hashlib
import json

import pytest

import train_quote_grpo_oof as oof


def provenance():
    return {
        "smoke_only": False, "prompt_tokens_match": True, "frozen_sft_reference": True,
        "baseline_sha256": "baseline", "runtime": {"torch": "fixed"},
        "training_arguments": dict(oof.FIXED_ARGUMENTS),
    }


def test_same_fixed_experiment_is_accepted():
    oof.verify_provenance(provenance(), "baseline", {"torch": "fixed"})


@pytest.mark.parametrize("key,value", [
    ("smoke_only", True), ("prompt_tokens_match", False), ("frozen_sft_reference", False),
    ("baseline_sha256", "changed"), ("runtime", {"torch": "different"}),
])
def test_oof_rejects_changed_runtime_or_inputs(key, value):
    with pytest.raises(ValueError, match="provenance"):
        oof.verify_provenance({**provenance(), key: value}, "baseline", {"torch": "fixed"})


@pytest.mark.parametrize("key,value", [
    ("learning_rate", 2e-5), ("num_train_epochs", 2), ("beta", 0),
    ("num_generations", 8), ("seed", 37), ("mask_truncated_completions", False),
])
def test_oof_cannot_retune_hyperparameters_after_pilot(key, value):
    record = copy.deepcopy(provenance())
    record["training_arguments"][key] = value
    with pytest.raises(ValueError, match="hyperparameter"):
        oof.verify_provenance(record, "baseline", {"torch": "fixed"})


def test_source_pin_normalizes_line_endings_but_rejects_code_changes(tmp_path, monkeypatch):
    monkeypatch.setattr(oof, "FROZEN_SOURCES", ("train.py",))
    source = tmp_path / "train.py"
    source.write_bytes(b"original\r\n")
    (tmp_path / "source_manifest.json").write_text(json.dumps({
        "files": {"train.py": hashlib.sha256(b"original\n").hexdigest()},
    }))
    oof.verify_sources(tmp_path, tmp_path)
    source.write_bytes(b"changed\n")
    with pytest.raises(ValueError, match="source changed"):
        oof.verify_sources(tmp_path, tmp_path)


def test_pilot_request_requires_non_smoke_fold_zero(tmp_path):
    request = {
        "script": "train_quote_grpo.py",
        "arguments": ["--baseline", str(tmp_path), "--sft-directory", str(tmp_path)],
    }
    assert oof.pilot_arguments(request, tmp_path).fold == 0
    for extra in (["--smoke"], ["--fold", "1"], ["--seed", "37"]):
        with pytest.raises(ValueError, match="Pilot request"):
            oof.pilot_arguments({**request, "arguments": request["arguments"] + extra}, tmp_path)
