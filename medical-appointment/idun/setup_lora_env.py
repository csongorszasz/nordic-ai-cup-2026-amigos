"""Create a run-local training overlay without upgrading the serving environment."""

import importlib.metadata
import json
import subprocess
import sys
import venv
from pathlib import Path


def main():
    root = Path(__file__).resolve().parent.parent
    target = root / ".lora-env"
    if target.exists():
        raise FileExistsError("Training overlay already exists; use a fresh isolated run.")
    protected = {
        name: importlib.metadata.version(name)
        for name in ("torch", "transformers", "faster-whisper", "ctranslate2", "huggingface-hub")
    }
    constraints = root / "results" / "serving_constraints.txt"
    constraints.write_text("\n".join(f"{name}=={version}" for name, version in protected.items()) + "\n")
    venv.EnvBuilder(with_pip=True, system_site_packages=True).create(target)
    python = target / "bin" / "python"
    subprocess.run(
        [str(python), "-m", "pip", "install", "--disable-pip-version-check",
         "-r", str(root / "requirements-lora.txt"), "-c", str(constraints)],
        check=True,
    )
    probe = (
        "import json,peft,accelerate;"
        "from importlib.metadata import version;"
        f"names={list(protected)!r};"
        "print(json.dumps({name:version(name) for name in names}))"
    )
    result = subprocess.run([str(python), "-c", probe], check=True, capture_output=True, text=True)
    observed = json.loads(result.stdout.strip().splitlines()[-1])
    if observed != protected:
        raise RuntimeError(f"Protected runtime changed in the training overlay: {observed!r}")
    manifest = {
        "python": str(python), "protected_runtime": protected,
        "packages": {"peft": "0.21.0", "accelerate": "1.15.0"},
        "serving_environment_modified": False, "complete": True,
    }
    (root / "results" / "lora_environment.json").write_text(json.dumps(manifest, indent=2))
    print(json.dumps(manifest, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
