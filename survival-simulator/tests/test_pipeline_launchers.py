import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from src.benchmarking.config import PROJECT_ROOT


def bash_path():
    found = shutil.which("bash")
    if found:
        return found
    git = shutil.which("git")
    candidate = Path(git).parent.parent / "bin" / "bash.exe" if git else None
    return str(candidate) if candidate and candidate.is_file() else None


@unittest.skipUnless(bash_path(), "Bash unavailable for bounded, mocked launcher tests")
class LauncherTests(unittest.TestCase):
    def run_launcher(self, extra, directory):
        startup = directory / "mocks.sh"
        trace = directory / "trace.txt"
        startup.write_text(
            'ssh() { printf "%s\\n" "$*" >> "$PIPELINE_TRACE"; '
            'case "$*" in *"sbatch --account"*) echo "Submitted batch job 42";; esac; return 0; }\n'
            'rsync() { echo "Unexpected sync of an existing frozen release" >&2; return 97; }\n'
            'python() { "$PIPELINE_PYTHON" "$@"; }\n',
            encoding="utf-8",
        )
        environment = {**os.environ, "BASH_ENV": startup.as_posix(),
                       "PIPELINE_TRACE": trace.as_posix(), "PIPELINE_PYTHON": Path(sys.executable).as_posix(),
                       "REMOTE_DIR": "/synthetic/nordic", "REMOTE": "mock-only"}
        result = subprocess.run(
            [bash_path(), (PROJECT_ROOT / "idun" / "submit.sh").as_posix(),
             "train", "configs/ppo-gru.json", "fixture", *extra],
            env=environment, capture_output=True, text=True, timeout=30,
        )
        return result, trace.read_text() if trace.exists() else ""

    def test_no_plan_stops_before_even_mock_ssh(self):
        with tempfile.TemporaryDirectory() as temporary:
            result, calls = self.run_launcher([], Path(temporary))
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("run-plan", result.stdout)
            self.assertEqual(calls, "")

    def test_reviewed_plan_uses_frozen_release_without_silent_worker_override(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            plan = directory / "plan.json"
            plan.write_text(json.dumps({"plan_id": "a" * 64}), encoding="utf-8")
            result, calls = self.run_launcher(
                ["--run-plan", plan.as_posix(), "--set", "resources.workers=3"], directory,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn("/releases/" + "a" * 64, calls)
            self.assertIn("--run-plan run-plan.json", calls)
            self.assertIn("resources.workers=3", calls)
            self.assertNotIn("rsync", calls)
            script = (PROJECT_ROOT / "idun" / "job_train.slurm").read_text()
            self.assertNotIn("resources.workers=", script)
            self.assertNotIn("resources.device=", script)


if __name__ == "__main__":
    unittest.main()
