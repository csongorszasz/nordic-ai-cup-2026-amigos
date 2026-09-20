"""Spoken-number preparation cannot upgrade the protected model runtime."""

import json
from types import SimpleNamespace

import pytest

from idun import setup_acoustic_env as setup


def fake_install(monkeypatch, root, *, wrong_version=False):
    calls = []
    monkeypatch.setattr(setup.importlib.metadata, "version", lambda name: "protected-" + name)

    def run(command, **options):
        calls.append(command)
        assert options["check"] is True
        if "-m" in command:
            assert "--no-deps" in command
            assert command[command.index("--target") + 1] == str(root / ".acoustic-deps")
            (root / ".acoustic-deps").mkdir()
            return SimpleNamespace(returncode=0)
        compile(command[-1], "<acoustic-dependency-probe>", "exec")
        payload = {
            "packages": {**setup.PACKAGES, **({"num2words": "wrong"} if wrong_version else {})},
            "renderings": setup.RENDERINGS,
            "module_file": str(root / ".acoustic-deps" / "num2words" / "__init__.py"),
        }
        return SimpleNamespace(stdout=json.dumps(payload))

    monkeypatch.setattr(setup.subprocess, "run", run)
    return calls


def test_preparation_installs_only_into_its_new_target(monkeypatch, tmp_path):
    calls = fake_install(monkeypatch, tmp_path)
    manifest = setup.prepare(tmp_path)
    assert len(calls) == 2
    assert manifest["complete"] is True and manifest["serving_environment_modified"] is False
    assert manifest["packages"] == setup.PACKAGES
    assert manifest["protected_runtime"] == {name: "protected-" + name for name in setup.PROTECTED}
    with pytest.raises(FileExistsError, match="fresh"):
        setup.prepare(tmp_path)


def test_unexpected_dependency_version_fails_explicitly(monkeypatch, tmp_path):
    fake_install(monkeypatch, tmp_path, wrong_version=True)
    with pytest.raises(RuntimeError, match="probe changed"):
        setup.prepare(tmp_path)
