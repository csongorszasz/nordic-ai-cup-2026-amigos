"""The serving launcher must forward thinking/prompt controls verbatim.

Runs ``idun/submit.sh serve`` with fake ``ssh``/``rsync`` on PATH and inspects
the constructed remote command. No network or cluster access.
"""

import os
import stat
import subprocess
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SUBMIT = PROJECT_ROOT / "idun" / "submit.sh"


def _fake(bin_dir: Path, name: str, body: str) -> None:
    path = bin_dir / name
    path.write_text("#!/bin/bash\n" + body)
    path.chmod(path.stat().st_mode | stat.S_IEXEC)


def test_serve_forwards_thinking_prompt_controls(tmp_path):
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    log = tmp_path / "ssh.log"
    _fake(bin_dir, "ssh", 'printf "%s\\n" "$*" >> "$FAKE_SSH_LOG"\n'
                          'case "$*" in *sbatch*) echo "Submitted batch job 12345";; esac\nexit 0\n')
    _fake(bin_dir, "rsync", "exit 0\n")

    env = dict(os.environ)
    env["PATH"] = f"{bin_dir}:{env['PATH']}"
    env["FAKE_SSH_LOG"] = str(log)
    env["MEDAPP_LLM_ENABLE_THINKING"] = "0"
    env["MEDAPP_LLM_REASONING_EFFORT"] = "low"
    env["MEDAPP_LLM_PROMPT"] = "audit"

    result = subprocess.run(
        ["bash", str(SUBMIT), "serve"], cwd=PROJECT_ROOT, env=env,
        capture_output=True, text=True, timeout=120,
    )
    assert result.returncode == 0, result.stderr
    command = log.read_text()
    assert "MEDAPP_LLM_ENABLE_THINKING=0" in command
    assert "MEDAPP_LLM_REASONING_EFFORT=low" in command
    assert "MEDAPP_LLM_PROMPT=audit" in command
