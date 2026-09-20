"""Offline two-speaker diarization for the supplied consultations.

Development-only: reads the local audio + transcripts, writes a sidecar under
``results/diarization/`` and prints a review table. Never used by ``/predict``.

    python diarize_conversations.py --tids sample_4 sample_20 sample_33 --print
    python diarize_conversations.py --limit 39
"""

import argparse
import json
import logging
from pathlib import Path

from answerers import diarize
from answerers.modernbert_data import load_transcript

PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_AUDIO = PROJECT_ROOT / "data" / "audio"
DEFAULT_OUT = PROJECT_ROOT / "results" / "diarization"


def review_table(tid: str, transcript, sidecar, audio_path: Path) -> str:
    lines = [
        f"===== {tid}  audio={audio_path.name}  "
        f"separation={sidecar['separation']:.3f}  role_margin={sidecar['role_margin']:.3f} ====="
    ]
    for seg in transcript.get("segments", []):
        entry = sidecar["segments"].get(str(seg["id"]), {})
        speaker = entry.get("speaker", "?")
        confidence = entry.get("confidence", 0.0)
        lines.append(
            f"  [s{seg['id']:02d} {seg['start']:6.2f}-{seg['end']:6.2f}] "
            f"{speaker:<7}({confidence:.2f})  {seg['text'][:88]}"
        )
    speakers = [entry["speaker"] for entry in sidecar["segments"].values()]
    turns = sum(1 for i, s in enumerate(speakers) if i == 0 or s != speakers[i - 1])
    labeled = sum(1 for s in speakers if s in ("doctor", "patient"))
    lines.append(
        f"  turns={turns}  labeled={labeled}/{len(speakers)}  "
        f"mixed={speakers.count('mixed')}"
    )
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tids", nargs="*", default=None, help="Transcript ids to process.")
    parser.add_argument("--limit", type=int, default=None, help="First N conversations.")
    parser.add_argument("--audio-dir", type=Path, default=DEFAULT_AUDIO)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--model", default=diarize.SPEAKER_MODEL)
    parser.add_argument("--revision", default=diarize.SPEAKER_REVISION)
    parser.add_argument("--print", action="store_true", help="Print a review table.")
    parser.add_argument("--require-complete", action=argparse.BooleanOptionalAction,
                        default=True,
                        help="Fail if any requested conversation could not be diarized.")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)

    if args.tids:
        tids = list(args.tids)
    else:
        from answerers.modernbert_data import load_rows

        tids = []
        for row in load_rows():
            if row["transcript_id"] not in tids:
                tids.append(row["transcript_id"])
    if args.limit:
        tids = tids[:args.limit]

    args.out.mkdir(parents=True, exist_ok=True)
    written, failed, tagged = 0, [], {"tagged": 0, "mixed": 0, "unknown": 0}
    for tid in tids:
        audio_path = args.audio_dir / f"conversation_{tid}.mp3"
        if not audio_path.exists():
            print(f"  {tid}: missing audio {audio_path}")
            failed.append(tid)
            continue
        transcript = load_transcript(tid)
        sidecar = diarize.run(
            transcript, str(audio_path), model_name=args.model, revision=args.revision
        )
        (args.out / f"{tid}.json").write_text(json.dumps(sidecar, indent=2))
        written += 1
        for entry in sidecar["segments"].values():
            speaker = entry.get("speaker")
            key = speaker if speaker in ("doctor", "patient") else (
                "mixed" if speaker == "mixed" else "unknown")
            tagged[key] += 1
        if args.print:
            print(review_table(tid, transcript, sidecar, audio_path))
    print(f"\nwrote {written}/{len(tids)} sidecars to {args.out}")
    print(f"segment labels: tagged={tagged['tagged']} mixed={tagged['mixed']} "
          f"unknown={tagged['unknown']}")
    if failed:
        print(f"failed conversations: {failed}")
        if args.require_complete:
            return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
