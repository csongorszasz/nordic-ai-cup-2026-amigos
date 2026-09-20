"""Install only spoken-number helpers in a run-local target, never serving."""

import importlib.metadata
import json
import subprocess
import sys
from pathlib import Path


PACKAGES = {"num2words": "0.5.14", "docopt": "0.6.2"}
PROTECTED = ("torch", "torchaudio", "transformers", "faster-whisper", "ctranslate2", "huggingface-hub")
RENDERINGS = {"100": "one hundred", "0.5": "zero point five", "-2": "minus two"}


def prepare(root):
    target = root / ".acoustic-deps"
    if target.exists():
        raise FileExistsError("Acoustic dependencies already exist; use a fresh isolated run.")
    protected = {name: importlib.metadata.version(name) for name in PROTECTED}
    subprocess.run([
        sys.executable, "-m", "pip", "install", "--disable-pip-version-check", "--quiet",
        "--no-deps", "--target", str(target), "-r", str(root / "requirements-acoustic.txt"),
    ], check=True)
    probe = (
        "import sys,json;from pathlib import Path;from importlib.metadata import version;"
        f"sys.path.insert(0,{str(target)!r});"
        "import num2words;"
        f"assert Path(num2words.__file__).resolve().is_relative_to(Path({str(target)!r}).resolve());"
        f"print(json.dumps({{'packages':{{name:version(name) for name in {list(PACKAGES)!r}}},"
        f"'renderings':{{value:num2words.num2words(value,lang='en') for value in {list(RENDERINGS)!r}}},"
        "'module_file':num2words.__file__}))"
    )
    result = subprocess.run([sys.executable, "-c", probe], capture_output=True, text=True, check=True)
    observed = json.loads(result.stdout.strip().splitlines()[-1])
    if observed["packages"] != PACKAGES or observed["renderings"] != RENDERINGS:
        raise RuntimeError(f"The spoken-number dependency probe changed: {observed}")
    if {name: importlib.metadata.version(name) for name in PROTECTED} != protected:
        raise RuntimeError("A protected runtime package changed during dependency preparation.")
    if not Path(observed["module_file"]).resolve().is_relative_to(target.resolve()):
        raise RuntimeError("The spoken-number helper did not load from the isolated target.")
    return {
        "dependency_root": str(target), "python": sys.executable,
        "packages": PACKAGES, "renderings": RENDERINGS, "protected_runtime": protected,
        "serving_environment_modified": False, "complete": True,
    }


def main():
    root = Path(__file__).resolve().parent.parent
    manifest = prepare(root)
    output = root / "results" / "acoustic_environment.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(manifest, indent=2))
    print(json.dumps(manifest, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
