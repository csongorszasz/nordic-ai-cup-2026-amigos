"""ASR smoke test: transcribe a few supplied conversations and show the evidence.

Run on an IDUN GPU node (see idun/job_smoke.slurm):

    python idun/asr_smoke.py sample_4 sample_17 sample_10

For each sample it prints the transcript with timestamps, then the gold questions
for that conversation, so the transcript can be judged against what the questions
actually need. Transcripts are cached to transcripts/<name>.json.
"""

import json
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from utils import (  # noqa: E402
    audio_duration_seconds,
    group_questions_by_conversation,
    load_sample_audio,
    audio_filename_for_transcript,
)

TRANSCRIPTS_DIR = PROJECT_ROOT / "transcripts"


def pick_groups(wanted):
    groups = group_questions_by_conversation()
    if not wanted:
        return groups[:3]
    wanted_files = {audio_filename_for_transcript(s) for s in wanted}
    chosen = [(f, rows) for f, rows in groups if f in wanted_files]
    missing = wanted_files - {f for f, _ in chosen}
    if missing:
        print(f"WARNING: no such sample(s): {sorted(missing)}")
    return chosen


def transcribe(model, audio_bytes, name):
    import tempfile

    started = time.time()
    with tempfile.NamedTemporaryFile(suffix=".mp3") as f:
        f.write(audio_bytes)
        f.flush()
        segments, info = model.transcribe(
            f.name,
            language="en",
            word_timestamps=True,
            vad_filter=True,
            condition_on_previous_text=False,
        )
        # segments is a generator; materialise it before the temp file closes.
        segments = [
            {"start": round(s.start, 2), "end": round(s.end, 2), "text": s.text.strip()}
            for s in segments
        ]
    elapsed = time.time() - started

    duration = audio_duration_seconds(audio_bytes) or 0.0
    rtf = duration / elapsed if elapsed else 0.0

    path = TRANSCRIPTS_DIR / f"{name}.json"
    path.write_text(json.dumps({"duration": duration, "segments": segments}, indent=2))

    return segments, info, elapsed, duration, rtf, path


def main() -> int:
    wanted = sys.argv[1:]
    groups = pick_groups(wanted)
    if not groups:
        print("No samples to transcribe.")
        return 1

    import torch
    from faster_whisper import WhisperModel

    device = "cuda" if torch.cuda.is_available() else "cpu"
    compute_type = "float16" if device == "cuda" else "int8"
    print(f"loading large-v3 (device={device}, compute_type={compute_type})...")
    load_started = time.time()
    model = WhisperModel("large-v3", device=device, compute_type=compute_type)
    print(f"model loaded in {time.time() - load_started:.1f}s")

    TRANSCRIPTS_DIR.mkdir(exist_ok=True)

    for audio_filename, rows in groups:
        name = audio_filename.replace("conversation_", "").replace(".mp3", "")
        print("\n" + "=" * 78)
        print(f"{audio_filename}   ({len(rows)} questions)")
        print("=" * 78)

        audio_bytes = load_sample_audio(audio_filename)
        segments, info, elapsed, duration, rtf, path = transcribe(
            model, audio_bytes, name
        )

        print(
            f"duration {duration:.1f}s  |  transcribe {elapsed:.1f}s  |  "
            f"RTF {rtf:.1f}x realtime  |  {len(segments)} segments  |  "
            f"lang {info.language} p={info.language_probability:.2f}"
        )
        print(f"cached -> {path.relative_to(PROJECT_ROOT)}")
        print("-" * 78)
        for s in segments:
            print(f"  [{s['start']:6.2f} - {s['end']:6.2f}]  {s['text']}")
        print("-" * 78)
        print("GOLD QUESTIONS")
        for row in rows:
            span = ""
            if row.get("evidence_start"):
                span = f"  evidence {row['evidence_start']}-{row['evidence_end']}"
            print(
                f"  {row['answer']:<3} {row['question_type']:<14} "
                f"{row['question']}{span}"
            )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
