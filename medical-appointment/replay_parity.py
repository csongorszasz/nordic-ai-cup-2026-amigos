"""Replay captured validation audio through the local pipeline for parity checks.

Development-only. Sets ``MEDAPP_CAPTURE=0`` before importing the pipeline so it
never overwrites the endpoint's stored responses. Run twice with different
``MEDAPP_SPAN_CALIBRATION`` settings, then compare the resulting spans against
the endpoint's stored captures.

    python replay_parity.py run --tag raw --convs sample_3,sample_80
    MEDAPP_SPAN_CALIBRATION=calibration/span_offset_base.json \
        python replay_parity.py run --tag cal --convs sample_3,sample_80
    python replay_parity.py compare --tags raw cal
"""

import argparse
import base64
import json
import os
import statistics
from pathlib import Path

os.environ["MEDAPP_CAPTURE"] = "0"

from dtos import ASRQuestionRequestDto  # noqa: E402
import example  # noqa: E402

ROOT = Path(__file__).resolve().parent
CAP = ROOT / "captured"
OUT = ROOT / "results" / "replay_parity"


def run(tag: str, convs) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    records = {}
    calibration = os.environ.get("MEDAPP_SPAN_CALIBRATION")
    for conv in convs:
        stem = f"conversation_{conv}"
        stored = json.loads((CAP / f"{stem}.json").read_text())
        audio = (CAP / "audio" / f"{stem}.mp3").read_bytes()
        request = ASRQuestionRequestDto(
            audio_base64=base64.b64encode(audio).decode(),
            audio_filename=f"{stem}.mp3",
            questions=stored["questions"],
        )
        response = example.predict(request)
        records[conv] = {
            "answers": response.answers,
            "evidence_start": response.evidence_start,
            "evidence_end": response.evidence_end,
        }
        print(f"[{tag}] {conv} calibration={calibration}")
        for i, (start, end) in enumerate(
            zip(response.evidence_start, response.evidence_end)
        ):
            keep = (stored["evidence_start"][i], stored["evidence_end"][i])
            print(f"  q{i}: replay=({start}, {end}) endpoint=({keep[0]}, {keep[1]})")
    (OUT / f"{tag}.json").write_text(json.dumps(records, indent=2))
    print("wrote", OUT / f"{tag}.json")


def compare(tags) -> None:
    data = {tag: json.loads((OUT / f"{tag}.json").read_text()) for tag in tags}
    for conv in data[tags[0]]:
        stored = json.loads((CAP / f"conversation_{conv}.json").read_text())
        print(f"== {conv} (endpoint minus replay; +0.2 start means calibration applied)")
        for tag in tags:
            start_deltas, end_deltas = [], []
            for i in range(len(stored["answers"])):
                s, e = stored["evidence_start"][i], stored["evidence_end"][i]
                ps = data[tag][conv]["evidence_start"][i]
                pe = data[tag][conv]["evidence_end"][i]
                if s is not None and ps is not None:
                    start_deltas.append(s - ps)
                if e is not None and pe is not None:
                    end_deltas.append(e - pe)
            mean_start = statistics.mean(start_deltas) if start_deltas else float("nan")
            mean_end = statistics.mean(end_deltas) if end_deltas else float("nan")
            max_start = max((abs(d) for d in start_deltas), default=float("nan"))
            print(
                f"  {tag:<4} n={len(start_deltas):>2} "
                f"mean_start={mean_start:+.3f} mean_end={mean_end:+.3f} "
                f"max|start|={max_start:.3f}"
            )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="mode", required=True)
    run_parser = sub.add_parser("run")
    run_parser.add_argument("--tag", required=True)
    run_parser.add_argument("--convs", default="sample_3,sample_80")
    cmp_parser = sub.add_parser("compare")
    cmp_parser.add_argument("--tags", nargs="+", default=["raw", "cal"])
    args = parser.parse_args()
    if args.mode == "run":
        run(args.tag, [c.strip() for c in args.convs.split(",") if c.strip()])
    else:
        compare(args.tags)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
