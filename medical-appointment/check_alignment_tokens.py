"""CPU-only tokenization preflight; synthetic times are not model predictions."""

import argparse
import csv
import hashlib
import json
import logging
from pathlib import Path

from answerers.retime import retime_words, word_unit_map
from benchmark import write_json


MODEL = "Qwen/Qwen3-ForcedAligner-0.6B-hf"
REVISION = "c07281df297b9905d24a508279258cccf987a064"
ROOT = Path(__file__).resolve().parent
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("results/alignment_tokens.json"))
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    requests = json.loads((args.baseline / "base_legacy_conversations.json").read_text())
    with (ROOT / "data" / "question_train.csv").open() as handle:
        expected = {row["transcript_id"] for row in csv.DictReader(handle)}
    if len(requests) != len(expected) or {entry["transcript_id"] for entry in requests} != expected:
        raise ValueError("Preflight requires all unique supplied training conversations.")
    transcripts = {}
    for entry in requests:
        transcript = json.loads(
            (args.baseline / "transcripts" / f"{entry['transcript_id']}.json").read_text()
        )
        digest = hashlib.sha256(json.dumps(transcript, sort_keys=True).encode()).hexdigest()
        if digest != entry["transcript_sha256"]:
            raise ValueError(f"Frozen transcript hash mismatch: {entry['transcript_id']}")
        transcripts[entry["transcript_id"]] = transcript
    from transformers import AutoProcessor

    processor = AutoProcessor.from_pretrained(MODEL, revision=REVISION, local_files_only=True)
    records = []
    for tid, transcript in transcripts.items():
        words = transcript["words"]
        text = "".join(word["word"] for word in words)
        tokens = processor.split_words_for_alignment(text, "English")
        try:
            mapping = word_unit_map(words, tokens)
            units = [
                {"text": token, "start_time": float(index), "end_time": float(index + 1)}
                for index, token in enumerate(tokens)
            ]
            retimed = retime_words(words, units, float(len(tokens)))
            if [word["word"] for word in retimed] != [word["word"] for word in words]:
                raise ValueError("Retiming changed original word text or ordering.")
        except ValueError as exc:
            logger.exception("Alignment token preflight failed for %s.", tid)
            records.append({"transcript_id": tid, "passed": False, "error": str(exc)})
            continue
        records.append({
            "transcript_id": tid, "passed": True, "original_words": len(words),
            "alignment_units": len(tokens), "punctuation_only_words": mapping.count(None),
            "multi_unit_words": sum(region is not None and region[0] != region[1] for region in mapping),
        })
    report = {
        "tokenization_preflight_only": True, "synthetic_times": True, "model": MODEL,
        "revision": REVISION, "timestamp_quantum_s": processor.timestamp_segment_time / 1000,
        "conversations": len(records), "passed": sum(row["passed"] for row in records),
        "records": records, "inference_network_access": False,
    }
    write_json(args.output, report)
    print(json.dumps({key: value for key, value in report.items() if key != "records"}), flush=True)
    return 0 if report["passed"] == len(records) else 1


if __name__ == "__main__":
    raise SystemExit(main())
