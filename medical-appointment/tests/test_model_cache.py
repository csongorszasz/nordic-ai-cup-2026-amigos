"""Model preparation is revision-pinned and explicitly size-bounded."""

from types import SimpleNamespace

import pytest

from idun.cache_model import planned_files


def test_manifest_filters_non_inference_artifacts():
    files = [
        SimpleNamespace(rfilename="model.safetensors", size=100),
        SimpleNamespace(rfilename="config.json", size=10),
        SimpleNamespace(rfilename="chat_template.jinja", size=5),
        SimpleNamespace(rfilename="training_data.bin", size=10000),
    ]
    assert sum(file["bytes"] for file in planned_files(files, 120)) == 115


def test_manifest_rejects_unknown_or_excessive_size():
    with pytest.raises(ValueError, match="Unknown"):
        planned_files([SimpleNamespace(rfilename="model.safetensors", size=None)], 100)
    with pytest.raises(ValueError, match="ceiling"):
        planned_files([SimpleNamespace(rfilename="model.safetensors", size=101)], 100)


def test_whisper_tokenizer_merges_are_included_but_training_pickles_are_not():
    files = [
        SimpleNamespace(rfilename="model.safetensors", size=100),
        SimpleNamespace(rfilename="merges.txt", size=5),
        SimpleNamespace(rfilename="vocab.json", size=5),
        SimpleNamespace(rfilename="training_args.bin", size=500),
    ]
    assert {entry["path"] for entry in planned_files(files, 110)} == {
        "model.safetensors", "merges.txt", "vocab.json",
    }
