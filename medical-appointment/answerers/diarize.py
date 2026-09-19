"""Lightweight two-speaker diarization for the medical consultations.

This is a **baseline**, not a state-of-the-art diarizer. It avoids pyannote and
torchaudio entirely: 16 kHz decoding comes from ``faster-whisper`` and speaker
embeddings from an open WavLM x-vector checkpoint, clustered with a
deterministic two-means. It targets the clean doctor/patient dyads of this case
and exists to test whether speaker-turn tags help evidence selection.

Pure helpers (window planning, clustering, role naming, assignment) stay
torch-free so the fast test suite can cover them; the heavy imports are lazy.
"""

import logging
import os
import re
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

logger = logging.getLogger(__name__)

SAMPLE_RATE = 16000
WINDOW_S = 1.5
HOP_S = 0.75
MIN_WINDOW_S = 0.2
MIXED_CONFIDENCE = 0.6
SPEAKER_MODEL = os.environ.get("MEDAPP_SPEAKER_MODEL", "microsoft/wavlm-base-plus-sv")
SPEAKER_REVISION = os.environ.get("MEDAPP_SPEAKER_REVISION") or None


_DOCTOR_RE = re.compile(
    r"\b(?:you|your|yours|yourself|let me|we(?:'| a)?ll|i(?:'| a)?ll|"
    r"recommend|prescribe|examine|listen|check|measure|breathe|chest|heart|"
    r"blood pressure|any|how|when|where|does|did|have you|are you|do you)\b",
    re.IGNORECASE,
)
_PATIENT_RE = re.compile(
    r"\b(?:i|i'?m|i'?ve|i'?d|my|mine|me|myself|we|our|pain|hurts?|aches?|"
    r"feel|feeling|symptom|problem|worried|worries|trouble)\b",
    re.IGNORECASE,
)
_QUESTION_RE = re.compile(r"\?")


def window_starts(
    segments: Sequence[Dict], duration: float,
    window_s: float = WINDOW_S, hop_s: float = HOP_S,
) -> List[float]:
    """Sliding-window starts that overlap speech, clipped to the audio.

    Windows let one Whisper segment that merges a speaker change still be split
    across both speakers, which whole-segment embeddings cannot do.
    """
    if duration <= 0:
        return []
    starts: List[float] = []
    cursor = 0.0
    end = min(duration, max((seg["end"] for seg in segments), default=0.0))
    while cursor + MIN_WINDOW_S <= end:
        if any(seg["start"] < cursor + window_s and seg["end"] > cursor for seg in segments):
            starts.append(round(cursor, 3))
        cursor += hop_s
    return starts


def two_means(embeddings: np.ndarray, iterations: int = 50) -> Tuple[np.ndarray, float]:
    """Deterministic two-means with an SVD split seed.

    Returns ``(labels, separation)`` where separation is the L2 distance
    between the two unit-normalised centroids: a small value means the audio
    looks single-speaker and the split is not trustworthy.
    """
    x = np.asarray(embeddings, dtype=np.float64)
    if x.ndim != 2 or x.shape[0] < 2:
        return np.zeros(x.shape[0], dtype=int), 0.0
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    z = x / norms
    centered = z - z.mean(axis=0, keepdims=True)
    _, _, vt = np.linalg.svd(centered, full_matrices=False)
    projection = centered @ vt[0]
    centers = np.stack([z[int(np.argmin(projection))], z[int(np.argmax(projection))]])
    labels: Optional[np.ndarray] = None
    for _ in range(iterations):
        distances = ((z[:, None, :] - centers[None, :, :]) ** 2).sum(axis=-1)
        new_labels = distances.argmin(axis=1)
        if labels is not None and np.array_equal(new_labels, labels):
            break
        labels = new_labels
        for cluster in (0, 1):
            members = labels == cluster
            if members.any():
                centers[cluster] = z[members].mean(axis=0)
    separation = float(np.linalg.norm(centers[0] - centers[1]))
    return labels, separation


def majority_label(labels: Sequence[int]) -> Tuple[int, float]:
    """Most common label and its share; ``(-1, 0.0)`` for no windows."""
    labels = list(labels)
    if not labels:
        return -1, 0.0
    counts = np.bincount(np.asarray(labels, dtype=int), minlength=2)
    winner = int(counts.argmax())
    return winner, float(counts[winner] / len(labels))


def median_smooth(labels: Sequence[int], width: int = 3) -> np.ndarray:
    """Odd-width median filter over window labels to remove single-window flips."""
    values = np.asarray(labels, dtype=int)
    if width <= 1 or values.size == 0:
        return values.copy()
    half = width // 2
    padded = np.pad(values, half, mode="edge")
    return np.array(
        [int(np.median(padded[i:i + width])) for i in range(values.size)],
        dtype=int,
    )


def assign_segments(
    segments: Sequence[Dict], starts: Sequence[float], labels: Sequence[int],
    window_s: float = WINDOW_S,
) -> Dict[int, Tuple[int, float]]:
    """Majority window label per segment, keyed by segment id."""
    buckets: Dict[int, List[int]] = {seg["id"]: [] for seg in segments}
    for start, label in zip(starts, labels):
        midpoint = start + window_s / 2.0
        for seg in segments:
            if seg["start"] <= midpoint <= seg["end"]:
                buckets[seg["id"]].append(int(label))
                break
    return {seg_id: majority_label(values) for seg_id, values in buckets.items()}


def _cluster_texts(
    segments: Sequence[Dict], assignment: Dict[int, Tuple[int, float]]
) -> Dict[int, List[str]]:
    texts: Dict[int, List[str]] = {0: [], 1: []}
    for seg in segments:
        cluster, _ = assignment.get(seg["id"], (-1, 0.0))
        if cluster in texts:
            texts[cluster].append(seg["text"])
    return texts


def role_scores(texts_by_cluster: Dict[int, List[str]]) -> Dict[int, float]:
    """Doctor-minus-patient lexical score per cluster (questions weigh double)."""
    scores: Dict[int, float] = {}
    for cluster, texts in texts_by_cluster.items():
        text = " ".join(texts)
        doctor = (
            2 * len(_QUESTION_RE.findall(text))
            + 2 * len(re.findall(r"\byour\b", text, re.IGNORECASE))
            + len(_DOCTOR_RE.findall(text))
        )
        patient = (
            2 * len(re.findall(r"\b(?:i|i'?m|i'?ve|i'?d|my|mine|me)\b", text, re.IGNORECASE))
            + len(_PATIENT_RE.findall(text))
        )
        scores[cluster] = float(doctor - patient)
    return scores


def name_roles(texts_by_cluster: Dict[int, List[str]]) -> Tuple[int, float]:
    """Which cluster is the doctor, and the normalised margin of that call."""
    scores = role_scores(texts_by_cluster)
    if not scores:
        return 0, 0.0
    ordered = sorted(scores.items(), key=lambda item: item[1], reverse=True)
    doctor_cluster = ordered[0][0]
    total = sum(abs(value) for value in scores.values()) or 1.0
    margin = abs(ordered[0][1] - ordered[-1][1]) / total
    return doctor_cluster, float(margin)


def decode_audio(path: str) -> np.ndarray:
    """Mono 16 kHz float32 waveform, decoded by faster-whisper's own reader."""
    from faster_whisper.audio import decode_audio

    return np.asarray(decode_audio(path, sampling_rate=SAMPLE_RATE), dtype=np.float32)


def embed_windows(
    waveform: np.ndarray, starts: Sequence[float],
    *, model_name: str = SPEAKER_MODEL, revision: Optional[str] = SPEAKER_REVISION,
    window_s: float = WINDOW_S, batch_size: int = 16,
) -> Tuple[np.ndarray, List[float]]:
    """WavLM x-vector per window; returns embeddings and the retained starts."""
    import torch
    from transformers import AutoProcessor, WavLMForXVector

    processor = AutoProcessor.from_pretrained(model_name, revision=revision)
    model = WavLMForXVector.from_pretrained(model_name, revision=revision)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device).eval()

    total = len(waveform)
    windows: List[np.ndarray] = []
    retained: List[float] = []
    for start in starts:
        first = int(round(start * SAMPLE_RATE))
        last = min(total, int(round((start + window_s) * SAMPLE_RATE)))
        if last - first < int(MIN_WINDOW_S * SAMPLE_RATE):
            continue
        windows.append(waveform[first:last])
        retained.append(start)

    chunks: List[np.ndarray] = []
    with torch.no_grad():
        for offset in range(0, len(windows), batch_size):
            batch = windows[offset:offset + batch_size]
            inputs = processor(
                batch, sampling_rate=SAMPLE_RATE, return_tensors="pt", padding=True
            )
            inputs = {key: value.to(device) for key, value in inputs.items()}
            output = model(**inputs)
            embeddings = getattr(output, "embeddings", None)
            if embeddings is None:
                embeddings = output[0]
            chunks.append(embeddings.detach().cpu().numpy())
    if not chunks:
        return np.zeros((0, 1), dtype=np.float64), retained
    return np.concatenate(chunks, axis=0), retained


def run(
    transcript: Dict, audio_path: str,
    *, model_name: str = SPEAKER_MODEL, revision: Optional[str] = SPEAKER_REVISION,
) -> Dict:
    """Diarize one conversation into a sidecar dict keyed by segment id."""
    waveform = decode_audio(audio_path)
    duration = len(waveform) / SAMPLE_RATE
    starts = window_starts(transcript.get("segments", []), duration)
    embeddings, starts = embed_windows(
        waveform, starts, model_name=model_name, revision=revision
    )
    labels, separation = two_means(embeddings)
    labels = median_smooth(labels)
    assignment = assign_segments(transcript.get("segments", []), starts, labels)
    doctor_cluster, margin = name_roles(_cluster_texts(transcript.get("segments", []), assignment))

    sidecar: Dict = {
        "method": "wavlm-xvector-2means",
        "model": model_name,
        "revision": revision,
        "separation": round(separation, 4),
        "role_margin": round(margin, 4),
        "segments": {},
        "words": [],
    }
    for seg in transcript.get("segments", []):
        cluster, confidence = assignment.get(seg["id"], (-1, 0.0))
        if cluster < 0 or confidence < MIXED_CONFIDENCE:
            speaker = "mixed"
        else:
            speaker = "doctor" if cluster == doctor_cluster else "patient"
        sidecar["segments"][str(seg["id"])] = {
            "speaker": speaker,
            "confidence": round(confidence, 3),
        }
    return sidecar


def attach(transcript: Dict, sidecar: Dict) -> Dict:
    """Copy segment-level speaker labels onto a transcript (in place)."""
    mapping = {int(seg_id): entry["speaker"] for seg_id, entry in sidecar["segments"].items()}
    for seg in transcript.get("segments", []):
        speaker = mapping.get(seg["id"])
        if speaker and speaker != "mixed":
            seg["speaker"] = speaker
    return transcript
