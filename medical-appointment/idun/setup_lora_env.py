"""Create a run-local training overlay without upgrading the serving environment."""

import argparse
import importlib.metadata
import json
import subprocess
import sys
import venv
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kind", choices=("lora", "grpo"), default="lora")
    args = parser.parse_args()
    root = Path(__file__).resolve().parent.parent
    target = root / f".{args.kind}-env"
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
         "-r", str(root / f"requirements-{args.kind}.txt"), "-c", str(constraints)],
        check=True,
    )
    training_packages = ["peft", "accelerate"]
    if args.kind == "grpo":
        training_packages.extend(["trl", "datasets"])
    probe = (
        "import json,peft,accelerate;"
        "from importlib.metadata import version;"
        f"names={list(protected) + training_packages!r};"
        "print(json.dumps({name:version(name) for name in names}))"
    )
    if args.kind == "grpo":
        probe = (
            "from trl import GRPOConfig,GRPOTrainer;"
            "from datasets import Dataset;"
            "GRPOConfig(output_dir='results/grpo_api_probe',use_cpu=True,bf16=False,"
            "per_device_train_batch_size=1,gradient_accumulation_steps=4,"
            "num_generations=4,beta=0.02,loss_type='dr_grpo',scale_rewards='none',"
            "report_to='none');"
        ) + probe
    result = subprocess.run([str(python), "-c", probe], check=True, capture_output=True, text=True)
    observed = json.loads(result.stdout.strip().splitlines()[-1])
    if {name: observed[name] for name in protected} != protected:
        raise RuntimeError(f"Protected runtime changed in the training overlay: {observed!r}")
    inherited = {name: importlib.metadata.version(name) for name in protected}
    if inherited != protected:
        raise RuntimeError(f"Shared serving environment was modified: {inherited!r}")
    manifest = {
        "python": str(python), "protected_runtime": protected,
        "packages": {name: observed[name] for name in training_packages},
        "serving_environment_modified": False, "complete": True,
    }
    (root / "results" / f"{args.kind}_environment.json").write_text(json.dumps(manifest, indent=2))
    print(json.dumps(manifest, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
