"""Replay training audio through the serving model with complete provenance.

Run on IDUN with idun/run.py. This does not contact the competition service.
The optional prompt/tokenization comparisons share exactly the same ASR cache.
"""

import argparse
import hashlib
import importlib.metadata
import json
import logging
import os
import random
import statistics
import time
from collections import defaultdict
from pathlib import Path

import asr
from answerers.llm import LLMAnswerer
from answerers.boundaries import adjusted_span
from answerers.llm_client import HFClient
from answerers.llm_prompt import VARIANTS
from dtos import ASRQuestionRequestDto
from example import predict
from llm_probe import score_records
from local_evaluator import Statistics
from utils import encode_audio, gold_evidence, group_questions_by_conversation, load_sample_audio

ROOT = Path(__file__).resolve().parent


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True), encoding="utf-8")
    temporary.replace(path)


class RecordingClient:
    def __init__(self, client):
        self.client = client
        self.calls = []
        self.model_name = client.model_name
        self.revision = client.revision

    def warm_up(self):
        self.client.warm_up()

    def generate(self, messages, **kwargs):
        started = time.monotonic()
        output = self.client.generate(messages, **kwargs)
        self.calls.append({
            "prompt_sha256": hashlib.sha256(
                json.dumps(messages, sort_keys=True).encode()
            ).hexdigest(),
            "raw_output": output,
            "generation_s": time.monotonic() - started,
        })
        return output


class RecordingAnswerer:
    def __init__(self, answerer):
        self.answerer = answerer
        self.infos = []
        self.transcript = None

    def answer_all(self, questions, transcript, *, deadline=None):
        self.transcript = transcript
        results = self.answerer.answer_all(
            questions, transcript, deadline=deadline, return_info=True
        )
        self.infos = [entry[2] for entry in results]
        return [(entry[0], entry[1]) for entry in results]


def paired_comparison(baseline, candidate, samples=2000, seed=13):
    """Conversation-resampled uncertainty for the exact composite-score delta."""
    left = {row["question_id"]: row for row in baseline}
    right = {row["question_id"]: row for row in candidate}
    if len(left) != len(baseline) or len(right) != len(candidate) or left.keys() != right.keys():
        raise ValueError("Comparison requires the same unique question IDs.")
    groups = defaultdict(lambda: [0, 0, 0, 0.0, 0.0, 0])
    for qid, row in left.items():
        other = right[qid]
        for field in ("transcript_id", "label", "gold", "question_type"):
            if row[field] != other[field]:
                raise ValueError(f"Mismatched {field} for {qid}.")
        bucket = groups[row["transcript_id"]]
        a, b = Statistics(), Statistics()
        a_iou = a.record(
            row["question_type"], row["label"], row["prediction"],
            row["gold"], row["span"],
        )
        b_iou = b.record(
            other["question_type"], other["label"], other["prediction"],
            other["gold"], other["span"],
        )
        bucket[0] += 1
        bucket[1] += a.correct
        bucket[2] += b.correct
        bucket[3] += a_iou
        bucket[4] += b_iou
        bucket[5] += len(a.tious)

    def delta(buckets):
        count = sum(item[0] for item in buckets)
        positives = sum(item[5] for item in buckets)
        a, b = Statistics(), Statistics()
        a.total = b.total = count
        a.correct = sum(item[1] for item in buckets)
        b.correct = sum(item[2] for item in buckets)
        if positives:
            a.tious = [sum(item[3] for item in buckets) / positives]
            b.tious = [sum(item[4] for item in buckets) / positives]
        return b.final_score - a.final_score

    buckets = list(groups.values())
    if not buckets:
        raise ValueError("Cannot compare empty runs.")
    rng = random.Random(seed)
    draws = sorted(
        delta(rng.choices(buckets, k=len(buckets))) for _ in range(samples)
    )
    return {
        "score_delta": delta(buckets),
        "conversation_bootstrap_95pct": [
            draws[int(samples * 0.025)], draws[min(samples - 1, int(samples * 0.975))]
        ],
        "conversations": len(buckets),
        "seed": seed,
        "bootstrap_samples": samples,
    }


def compare_reference(
    baseline, candidate, baseline_requests, candidate_requests, *, allow_asr_change=False
):
    left = {row["transcript_id"]: row for row in baseline_requests}
    right = {row["transcript_id"]: row for row in candidate_requests}
    if left.keys() != right.keys():
        raise ValueError("Reference and candidate cover different conversations.")
    changed_transcripts = []
    for tid in left:
        if left[tid]["audio_sha256"] != right[tid]["audio_sha256"]:
            raise ValueError(f"Unmatched audio_sha256 for {tid}.")
        if left[tid]["transcript_sha256"] != right[tid]["transcript_sha256"]:
            changed_transcripts.append(tid)
            if not allow_asr_change:
                raise ValueError(f"Unmatched transcript_sha256 for {tid}.")
    excluded = {
        tid for request in [*baseline_requests, *candidate_requests]
        for tid in request["demonstration_tids"]
    }
    result = {
        "excluded_demonstration_tids": sorted(excluded),
        "matched_audio": True, "matched_transcripts": not changed_transcripts,
        "changed_transcript_ids": sorted(changed_transcripts),
        "comparison_axis": "asr" if allow_asr_change else "answerer",
    }
    for name, offsets in (("raw", (0.0, 0.0)), ("fixed_start_0_2", (0.2, 0.0))):
        def apply(records):
            return [
                {
                    **row,
                    "span": adjusted_span(row["span"], offsets, row.get("duration"))
                    if row["answer"] else None,
                }
                for row in records
            ]
        base = apply(baseline)
        predicted = apply(candidate)
        safe_base = [row for row in base if row["transcript_id"] not in excluded]
        safe_predicted = [row for row in predicted if row["transcript_id"] not in excluded]
        result[name] = {
            "all_questions": {
                "baseline": score_records(base), "candidate": score_records(predicted),
                "paired": paired_comparison(base, predicted),
            },
            "demonstration_disjoint": {
                "baseline": score_records(safe_base), "candidate": score_records(safe_predicted),
                "paired": paired_comparison(safe_base, safe_predicted),
            },
        }
    return result


def run_configuration(client, conversations, variant, tokenization, output, force_asr=False):
    client.legacy_special_tokens = tokenization == "legacy"
    recorded_client = RecordingClient(client)
    answerer = LLMAnswerer(client=recorded_client, variant=variant)
    answerer.warm_up()
    wrapped = RecordingAnswerer(answerer)
    tag = f"{variant}_{tokenization}"
    records, requests = [], []
    all_count = sum(len(rows) for _, rows in conversations)
    write_json(output / f"{tag}_summary.json", {"complete": False, "expected_questions": all_count})
    for index, (filename, rows) in enumerate(conversations):
        audio = load_sample_audio(filename)
        cache = asr.cache_path(filename, audio)
        cache_hit = asr.CACHE_ENABLED and cache.exists()
        if force_asr:
            # Only remove this isolated run's cache, never the shared incumbent's.
            cache.unlink(missing_ok=True)
            cache_hit = False
        recorded_client.calls.clear()
        wrapped.infos = []
        wrapped.transcript = None
        request = ASRQuestionRequestDto(
            audio_base64=encode_audio(audio), audio_filename=filename,
            questions=[row["question"] for row in rows],
        )
        started = time.monotonic()
        response = predict(request, answerer=wrapped)
        elapsed = time.monotonic() - started
        transcript_id = rows[0]["transcript_id"]
        transcript = wrapped.transcript
        transcript_hash = None
        if transcript is not None:
            transcript_hash = hashlib.sha256(
                json.dumps(transcript, sort_keys=True).encode()
            ).hexdigest()
            write_json(output / "transcripts" / f"{transcript_id}.json", transcript)
        requests.append({
            "transcript_id": transcript_id, "audio_filename": filename,
            "audio_sha256": hashlib.sha256(audio).hexdigest(),
            "transcript_sha256": transcript_hash, "asr_cache_hit": cache_hit,
            "asr_config_hash": asr.config_hash(), "latency_s": elapsed,
            "demonstration_tids": list(answerer.few_shot_sources),
            "calls": list(recorded_client.calls),
        })
        for question_index, row in enumerate(rows):
            info = wrapped.infos[question_index] if question_index < len(wrapped.infos) else {}
            answer = response.answers[question_index]
            span = (
                [response.evidence_start[question_index], response.evidence_end[question_index]]
                if answer else None
            )
            gold = gold_evidence(row)
            records.append({
                "question_id": row["question_id"], "transcript_id": transcript_id,
                "question": row["question"], "question_type": row["question_type"],
                "label": int(row["label"]), "answer": answer, "prediction": int(answer),
                "span": span, "gold": list(gold) if gold is not None else None,
                "quote": info.get("quote"), "word_range": info.get("word_range"),
                "raw_answer": info.get("raw_answer", answer),
                "decided_by": info.get("decided_by", "pipeline_fallback"),
                "proposed_span": info.get("span", span),
                "duration": transcript.get("duration") if transcript is not None else None,
                "original_span": info.get("original_span", span),
                "calibration_applied": info.get("calibration_applied", False),
            })
        write_json(output / f"{tag}_questions.json", records)
        write_json(output / f"{tag}_conversations.json", requests)
        print(
            f"{tag} [{index + 1}/{len(conversations)}] {transcript_id} "
            f"{elapsed:.2f}s cache={cache_hit} yes={sum(response.answers)}",
            flush=True,
        )
    summary = score_records(records)
    latencies = sorted(item["latency_s"] for item in requests)
    summary.update({
        "complete": len(records) == all_count, "expected_questions": all_count,
        "full_corpus": all_count == sum(
            len(rows) for _, rows in group_questions_by_conversation()
        ),
        "model": client.model_name, "revision": client.revision,
        "span_calibration": answerer.calibration.metadata() if answerer.calibration else None,
        "variant": variant, "tokenization": tokenization,
        "num_beams": client.num_beams,
        "asr_model": asr.MODEL_SIZE, "asr_config_hash": asr.config_hash(),
        "parse_failures": sum(row["decided_by"] == "parse" for row in records),
        "alignment_failures": sum(row["decided_by"] == "alignment" for row in records),
        "pipeline_fallbacks": sum(row["decided_by"] == "pipeline_fallback" for row in records),
        "latency_mean_s": statistics.mean(latencies),
        "latency_p95_s": latencies[min(len(latencies) - 1, int(0.95 * len(latencies)))],
        "latency_max_s": max(latencies),
        "cache_hits": sum(item["asr_cache_hit"] for item in requests),
    })
    write_json(output / f"{tag}_summary.json", summary)
    print(json.dumps(summary, sort_keys=True), flush=True)
    return records


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--variants", nargs="+", choices=VARIANTS, default=["base"])
    parser.add_argument("--tokenization", nargs="+", choices=("legacy", "template"), default=["legacy"])
    parser.add_argument("--limit", type=int)
    parser.add_argument("--force-asr", action="store_true")
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--output", default="results/benchmark")
    parser.add_argument("--baseline-records")
    parser.add_argument("--baseline-requests")
    parser.add_argument("--allow-asr-change", action="store_true")
    parser.add_argument("--expect-asr-compute-type")
    parser.add_argument("--expect-llm-dtype")
    args = parser.parse_args()
    if args.force_asr and not (ROOT / "run_request.json").exists():
        parser.error("--force-asr is restricted to isolated IDUN snapshots.")
    if bool(args.baseline_records) != bool(args.baseline_requests):
        parser.error("Both reference prediction and request manifests are required.")
    if args.allow_asr_change and not args.baseline_records:
        parser.error("--allow-asr-change requires a reference run.")
    logging.basicConfig(level=logging.INFO)
    output = Path(args.output)
    conversations = group_questions_by_conversation()
    if args.limit:
        conversations = conversations[:args.limit]
    if not conversations:
        parser.error("No conversations selected.")
    reference = (
        json.loads(Path(args.baseline_records).read_text()) if args.baseline_records else None
    )
    reference_requests = (
        json.loads(Path(args.baseline_requests).read_text()) if args.baseline_requests else None
    )
    client = HFClient(
        model_name=args.model, revision=args.revision,
        max_new_tokens=args.max_new_tokens,
        legacy_special_tokens=args.tokenization[0] == "legacy",
    )
    asr.warm_up()
    asr_runtime = {
        "compute_type": asr.get_model().model.compute_type,
        "device": asr.get_model().model.device,
        "device_index": asr.get_model().model.device_index,
    }
    if args.expect_asr_compute_type and asr_runtime["compute_type"] != args.expect_asr_compute_type:
        raise RuntimeError(f"Unexpected ASR runtime precision: {asr_runtime!r}")
    client.warm_up()
    import torch
    actual_llm_dtype = str(next(client._model.parameters()).dtype).removeprefix("torch.")
    if args.expect_llm_dtype and actual_llm_dtype != args.expect_llm_dtype:
        raise RuntimeError(f"Unexpected LLM parameter dtype: {actual_llm_dtype}")

    write_json(output / "environment.json", {
        "packages": {
            name: importlib.metadata.version(name)
            for name in ("torch", "transformers", "faster-whisper", "ctranslate2")
        },
        "gpu": torch.cuda.get_device_name() if torch.cuda.is_available() else "cpu",
        "llm": {
            "model": client.model_name, "revision": client.revision,
            "requested_dtype": client.dtype, "actual_dtype": actual_llm_dtype,
            "device": client._device,
            "num_beams": client.num_beams,
        },
        "arguments": vars(args),
        "asr": {"model": asr.MODEL_SIZE, "compute_type": asr.COMPUTE_TYPE,
                "config_hash": asr.config_hash(), "runtime": asr_runtime},
        "job_id": os.environ.get("SLURM_JOB_ID"),
    })
    baseline = None
    comparisons = {}
    for tokenization in args.tokenization:
        for variant in args.variants:
            records = run_configuration(
                client, conversations, variant, tokenization, output,
                args.force_asr and baseline is None,
            )
            if baseline is None:
                baseline = records
            else:
                comparisons[f"{variant}_{tokenization}"] = paired_comparison(baseline, records)
            if reference is not None:
                requests = json.loads((output / f"{variant}_{tokenization}_conversations.json").read_text())
                comparison = compare_reference(
                    reference, records, reference_requests, requests,
                    allow_asr_change=args.allow_asr_change,
                )
                write_json(output / f"{variant}_{tokenization}_reference.json", comparison)
    write_json(output / "comparisons.json", comparisons)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
