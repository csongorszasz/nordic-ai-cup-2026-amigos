"""Submit isolated IDUN experiments from Windows or Linux.

    python idun/run.py submit --tag baseline --gpu80 --script benchmark.py -- --limit 3
    python idun/run.py status --run-id baseline-<id>
    python idun/run.py wait --run-id baseline-<id>
    python idun/run.py pull --run-id baseline-<id>

Only training transcripts are copied. The live project and environment are
never synchronized or modified, and a remote lock prevents overlapping runs.
"""

import argparse
import base64
import csv
import hashlib
import io
import json
import re
import shlex
import subprocess
import sys
import tarfile
import time
import uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RUNS = ROOT / "results" / "idun_runs"
TEXT_SUFFIXES = {".py", ".sh", ".slurm", ".txt", ".ini", ".csv", ".json", ".md"}
SOURCE_DIRS = {"answerers", "verifier", "tests", "idun", "local", "annotations", "docs"}


def command(args, *, capture=True):
    return subprocess.run(
        args, check=True, text=True, encoding="utf-8",
        stdout=subprocess.PIPE if capture else None,
    ).stdout


def ssh(remote, script):
    return command([
        "ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=15",
        "-o", "StrictHostKeyChecking=yes", remote, script,
    ]).strip()


def snapshot(destination, request, assets=()):
    listed = command([
        "git", "-C", str(ROOT), "ls-files", "--cached", "--others",
        "--exclude-standard", "--", ".",
    ]).splitlines()
    manifest = {
        "commit": command(["git", "-C", str(ROOT), "rev-parse", "HEAD"]).strip(),
        "files": {},
    }
    with tarfile.open(destination, "w:gz") as archive:
        asset_names = {Path(name).as_posix() for name in assets}
        names = set(listed) | asset_names
        for name in sorted(names):
            path = ROOT / name
            parts = Path(name).parts
            include = (
                len(parts) == 1 and path.suffix in {".py", ".txt", ".ini", ".md"}
                or parts[0] in SOURCE_DIRS
                or name.replace("\\", "/") == "data/question_train.csv"
                or name in asset_names
            )
            if not include or not path.is_file() or path.suffix not in TEXT_SUFFIXES:
                continue
            content = path.read_bytes().replace(b"\r\n", b"\n")
            member_name = Path(name).as_posix()
            manifest["files"][member_name] = hashlib.sha256(content).hexdigest()
            info = tarfile.TarInfo(member_name)
            info.size = len(content)
            info.mode = 0o644
            archive.addfile(info, io.BytesIO(content))
        for name, payload in (
            ("source_manifest.json", manifest), ("run_request.json", request)
        ):
            content = json.dumps(payload, indent=2, sort_keys=True).encode()
            info = tarfile.TarInfo(name)
            info.size = len(content)
            info.mode = 0o644
            archive.addfile(info, io.BytesIO(content))
    return manifest


def allocation_arguments(*, cpu, gpu80, cpu_cores=None, cpu_memory_gb=None):
    if not cpu:
        if cpu_cores is not None or cpu_memory_gb is not None:
            raise ValueError("CPU resource overrides require --cpu.")
        constraint = "gpu80g" if gpu80 else "gpu32g|gpu40g|gpu80g"
        return f"--constraint={shlex.quote(constraint)} --mem={'128G' if gpu80 else '64G'}"
    if gpu80:
        raise ValueError("CPU allocations cannot request a GPU constraint.")
    cores = 2 if cpu_cores is None else cpu_cores
    memory = 8 if cpu_memory_gb is None else cpu_memory_gb
    if any(isinstance(value, bool) or not isinstance(value, int) or value < 1 for value in (cores, memory)):
        raise ValueError("CPU cores and memory must be positive integers.")
    return f"--mem={memory}G --cpus-per-task={cores}"


def submit(args):
    resource_args = allocation_arguments(
        cpu=args.cpu, gpu80=args.gpu80, cpu_cores=args.cpu_cores, cpu_memory_gb=args.cpu_memory_gb,
    )
    run_id = f"{args.tag}-{uuid.uuid4().hex[:8]}"
    local = RUNS / run_id
    local.mkdir(parents=True)
    environment = {}
    for assignment in args.env:
        key, separator, value = assignment.partition("=")
        if not separator or not key.startswith(("MEDAPP_", "WHISPER_", "NLI_")):
            raise ValueError(f"Unsupported experiment environment assignment: {key}")
        environment[key] = value
    script = Path(args.script)
    if script.is_absolute() or ".." in script.parts or not (ROOT / script).is_file():
        raise ValueError("Experiment script must be an existing project-relative file.")
    request = {
        "run_id": run_id, "script": script.as_posix(),
        "arguments": args.script_args,
        "environment": environment,
        "input_parent_run": args.reference_run,
        "role": args.role,
        "walltime": args.walltime,
        "cpu_only": args.cpu,
        "allocation_arguments": resource_args,
        "after_job": args.after_job,
    }
    if request["arguments"][:1] == ["--"]:
        request["arguments"] = request["arguments"][1:]
    archive = local / "source.tgz"
    assets = []
    for name in args.asset:
        path = Path(name)
        if (
            path.is_absolute() or ".." in path.parts or path.suffix != ".json"
            or not path.parts or path.parts[0] != "models"
            or not (ROOT / path).is_file()
        ):
            raise ValueError("Assets must be existing JSON artifacts under models.")
        assets.append(path)
    manifest = snapshot(archive, request, assets=assets)
    home = ssh(args.remote, 'printf "%s" "$HOME"')
    remote_root = f"{home}/nordic-medical-runs"
    remote_dir = f"{remote_root}/{run_id}"
    upload = f"{remote_root}/{run_id}.tgz"
    input_dir = f"{home}/nordic-medical"
    if args.reference_run:
        parent = load_run(args.reference_run)
        if parent["remote"] != args.remote:
            raise ValueError("Reference inputs must be on the same IDUN host.")
        input_dir = parent["remote_dir"]
    ssh(args.remote, f"mkdir -p {shlex.quote(remote_root)}")
    command(["scp", "-q", str(archive), f"{args.remote}:{upload}"], capture=False)

    stage = r'''
import csv, hashlib, json, shutil, sys
from pathlib import Path
root = Path.cwd()
shared = Path(sys.argv[1])
(root / "data" / "audio").mkdir()
(root / "transcripts").mkdir()
manifest = {"source": str(shared), "audio": {}, "transcripts": {}}
with (root / "data" / "question_train.csv").open() as handle:
    tids = {row["transcript_id"] for row in csv.DictReader(handle)}
for tid in sorted(tids):
    filename = f"conversation_{tid}.mp3"
    source = shared / "data" / "audio" / filename
    target = root / "data" / "audio" / filename
    shutil.copy2(source, target)
    audio_hash = hashlib.sha256(target.read_bytes()).hexdigest()
    manifest["audio"][filename] = audio_hash
    for path in (shared / "transcripts").glob(f"conversation_{tid}.*.json"):
        parts = path.name.split(".")
        if len(parts) == 4 and parts[2] != audio_hash[:8]:
            continue
        shutil.copy2(path, root / "transcripts" / path.name)
        manifest["transcripts"][path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
(root / "input_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True))
'''
    active_file = "active-cpu-job" if args.cpu else "active-job"
    prefix = "cpu" if args.cpu else "serve" if args.role == "serving" else "iter"
    job_script = "idun/job_cpu_experiment.slurm" if args.cpu else "idun/job_experiment.slurm"
    remote_command = f"""
set -eu
root={shlex.quote(remote_root)}
mkdir "$root/active.lock" || {{ echo 'Another submission holds the experiment lock.' >&2; exit 1; }}
trap 'rmdir "$root/active.lock"' EXIT
jobs=$(squeue --me --noheader --format=%i)
if test -f "$root/{active_file}"; then
    active=$(cat "$root/{active_file}")
    if printf '%s\\n' "$jobs" | grep -Fxq "$active"; then
        echo "Experiment $active is still active; refusing overlap." >&2
        exit 1
    fi
    medical=$(squeue --me --noheader --format='%i|%j|%b')
    count=$(printf '%s\\n' "$medical" | awk -F '|' '$2 ~ /^med_/ && $3 ~ /gpu/ {{n++}} END {{print n+0}}')
    dependency_ok=0
    if test -n "{args.after_job or ''}"; then
        dependency_ok=$(printf '%s\\n' "$medical" | awk -F '|' '$1 == "{args.after_job or ''}" && $2 ~ /^med_serve/ && $3 ~ /gpu/ {{n=1}} END {{print n+0}}')
        if test "$dependency_ok" -ne 1; then
            echo "The dependency must be an active owned medical serving allocation." >&2
            exit 1
        fi
    fi
    if test "{int(args.cpu)}" -eq 0 && {{ test "$count" -gt 2 || {{ test "$count" -eq 2 && test "$dependency_ok" -ne 1; }}; }}; then
        echo "Two medical GPU allocations already exist; refusing a third." >&2
        exit 1
    fi
fi
mkdir {shlex.quote(remote_dir)}
cd {shlex.quote(remote_dir)}
tar -xzf {shlex.quote(upload)}
rm -- {shlex.quote(upload)}
mkdir logs results
python3 -c {shlex.quote(stage)} {shlex.quote(input_dir)}
job=$(sbatch --parsable --account=share-ie-idi --job-name=med_{prefix}_{args.tag[:24]} \
    {resource_args} \
    {'--time=' + shlex.quote(args.walltime) if args.walltime else ''} \
    {'--dependency=afterany:' + args.after_job if args.after_job else ''} \
    {job_script})
job=${{job%%;*}}
printf '%s\\n' "$job" > "$root/{'serving-job' if args.role == 'serving' else active_file}"
printf '%s\\n' "$job"
"""
    job_id = ssh(args.remote, remote_command).splitlines()[-1]
    if not job_id.isdigit():
        raise ValueError(f"Unexpected SLURM job id: {job_id!r}")
    record = {
        **request, "job_id": job_id, "remote": args.remote,
        "remote_dir": remote_dir, "commit": manifest["commit"],
        "archive_sha256": hashlib.sha256(archive.read_bytes()).hexdigest(),
    }
    (local / "run.json").write_text(json.dumps(record, indent=2))
    print(json.dumps(record, indent=2), flush=True)


def load_run(run_id):
    if not run_id or not re.fullmatch(r"[a-zA-Z0-9_-]+", run_id):
        raise ValueError("A valid --run-id is required.")
    return json.loads((RUNS / run_id / "run.json").read_text())


def status(run):
    listing = ssh(
        run["remote"],
        "squeue --me --noheader --format='%i|%T|%M|%N'",
    )
    live = "\n".join(
        line for line in listing.splitlines()
        if line.split("|", 1)[0] == run["job_id"]
    )
    if live:
        return True, live
    finished = ssh(
        run["remote"],
        f"sacct -X --noheader --parsable2 -j {run['job_id']} "
        "--format=JobID,State,ExitCode,Elapsed",
    )
    return False, finished


def matches_expected(payload, expected):
    count = len(expected)
    for field in ("answers", "evidence_start", "evidence_end"):
        if not isinstance(payload.get(field), list) or len(payload[field]) != count:
            return False
    for index, record in enumerate(expected):
        answer = payload["answers"][index]
        if not isinstance(answer, bool) or answer != record["answer"]:
            return False
        actual = [payload["evidence_start"][index], payload["evidence_end"][index]]
        wanted = record["span"]
        if wanted is None:
            if actual != [None, None]:
                return False
        elif any(
            isinstance(value, bool) or not isinstance(value, (int, float))
            or not abs(value - target) < 1e-6
            for value, target in zip(actual, wanted)
        ):
            return False
    return True


def public_request(method, url, *, payload=None):
    import requests

    response = requests.request(
        method, url, json=payload, timeout=60 if payload is not None else 15,
        allow_redirects=False,
    )
    response.raise_for_status()
    return response.status_code, response.json()


def verify_public(run, pending):
    """Probe our endpoint externally, without a competition API or relaxed TLS."""
    public = pending["url"]
    if not re.fullmatch(r"https://[a-z0-9-]+\.trycloudflare\.com", public):
        raise ValueError("Unexpected candidate tunnel hostname.")
    root_status, root_body = public_request("GET", public + "/")
    if root_status != 200 or root_body != "Your endpoint is running!":
        raise ValueError("Public root is not the expected candidate application.")
    code = (
        "import json;from pathlib import Path;"
        f"r=Path({run['remote_dir']!r});"
        "rows=json.loads((r/'results/http_benchmark/questions.json').read_text());"
        "first=rows[0]['transcript_id'];"
        "selected=[row for row in rows if row['transcript_id']==first];"
        "manifest=json.loads((r/'input_manifest.json').read_text());"
        "print(json.dumps({'records':selected,'sha256':manifest['audio'][selected[0]['audio_filename']]}))"
    )
    proof = json.loads(ssh(run["remote"], "python3 -c " + shlex.quote(code)))
    expected = proof["records"]
    filename = expected[0]["audio_filename"]
    audio = (ROOT / "data" / "audio" / filename).read_bytes()
    if hashlib.sha256(audio).hexdigest() != proof["sha256"]:
        raise ValueError("Local probe audio differs from the qualified release.")
    with (ROOT / "data" / "question_train.csv").open(encoding="utf-8", newline="") as handle:
        questions = {row["question_id"]: row["question"] for row in csv.DictReader(handle)}
    payload = {
        "audio_filename": filename,
        "audio_base64": base64.b64encode(audio).decode(),
        "questions": [questions[row["question_id"]] for row in expected],
    }
    started = time.monotonic()
    predict_status, body = public_request(
        "POST", public + "/predict", payload=payload
    )
    elapsed = time.monotonic() - started
    if predict_status != 200 or not matches_expected(body, expected) or elapsed >= 58:
        raise ValueError("Public predictions or latency differ from the qualified release.")
    verification = {
        "url": public, "root_status": root_status, "predict_status": predict_status,
        "predictions_match": True, "question_count": len(expected),
        "latency_s": elapsed, "source": "external-controller",
    }
    target = f"{run['remote_dir']}/results/http_benchmark/external_verification.json"
    temporary = target + ".tmp"
    ssh(
        run["remote"],
        f"printf '%s' {shlex.quote(json.dumps(verification))} > {shlex.quote(temporary)} "
        f"&& mv -f {shlex.quote(temporary)} {shlex.quote(target)}",
    )
    return verification


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("submit", "status", "wait", "ready", "pull"))
    parser.add_argument("--remote", default="idun")
    parser.add_argument("--tag", default="experiment")
    parser.add_argument("--run-id")
    parser.add_argument("--reference-run")
    parser.add_argument("--role", choices=("experiment", "serving"), default="experiment")
    parser.add_argument("--walltime")
    parser.add_argument("--after-job")
    parser.add_argument("--gpu80", action="store_true")
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--cpu-cores", type=int)
    parser.add_argument("--cpu-memory-gb", type=int)
    parser.add_argument("--script")
    parser.add_argument("--asset", action="append", default=[])
    parser.add_argument("--env", action="append", default=[])
    args, script_args = parser.parse_known_args()
    args.script_args = script_args
    if not re.fullmatch(r"[a-zA-Z0-9_-]+", args.tag):
        parser.error("--tag must contain only letters, digits, hyphens, and underscores.")
    if args.walltime and not re.fullmatch(r"(?:\d+-)?\d{1,2}:\d{2}:\d{2}", args.walltime):
        parser.error("--walltime must use [days-]hours:minutes:seconds.")
    if args.cpu and (args.gpu80 or args.role == "serving"):
        parser.error("--cpu cannot be combined with --gpu80 or --role serving.")
    if args.after_job and (
        not args.after_job.isdigit() or args.cpu or args.role != "experiment"
    ):
        parser.error("--after-job is supported only for a GPU experiment and a numeric job ID.")
    if args.action == "submit":
        if not args.script:
            parser.error("submit requires --script.")
        submit(args)
        return 0
    run = load_run(args.run_id)
    if args.action == "ready":
        endpoint = f"{run['remote_dir']}/results/http_benchmark/endpoint.json"
        pending_path = f"{run['remote_dir']}/results/http_benchmark/publication_pending.json"
        verified = False
        while True:
            payload = ssh(
                run["remote"],
                f"if test -f {shlex.quote(endpoint)}; then cat {shlex.quote(endpoint)}; fi",
            )
            active, state = status(run)
            if not active:
                raise RuntimeError(f"Serving allocation ended before readiness: {state}")
            if payload:
                json.loads(payload)
                print(payload, flush=True)
                return 0
            pending = ssh(
                run["remote"],
                f"if test -f {shlex.quote(pending_path)}; then cat {shlex.quote(pending_path)}; fi",
            )
            if pending and not verified and time.time() >= json.loads(pending).get("verify_after", 0):
                try:
                    proof = verify_public(run, json.loads(pending))
                    print("External verification " + json.dumps(proof), flush=True)
                    verified = True
                except (ValueError, OSError) as exc:
                    print(f"Public verification pending: {type(exc).__name__}: {exc}", flush=True)
            print(state, flush=True)
            time.sleep(60)
    if args.action == "pull":
        target = RUNS / args.run_id / "artifacts"
        target.mkdir(exist_ok=True)
        for name in ("results", "logs"):
            command([
                "scp", "-q", "-r",
                f"{run['remote']}:{run['remote_dir']}/{name}", str(target),
            ], capture=False)
        print(str(target), flush=True)
        return 0
    while True:
        active, state = status(run)
        print(state or "SLURM accounting not yet available.", flush=True)
        if args.action == "status" or not active:
            break
        time.sleep(45)
    return 0


if __name__ == "__main__":
    sys.exit(main())
