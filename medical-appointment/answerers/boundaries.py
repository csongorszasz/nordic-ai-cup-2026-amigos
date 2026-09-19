"""Small, provenance-bound correction from word times to annotated boundaries."""

import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional, Sequence, Tuple


def adjusted_span(span, offsets, duration=None):
    """Keep the original citation when the proposed correction would collapse it."""
    if span is None:
        return None
    start = max(0.0, span[0] + offsets[0])
    end = span[1] + offsets[1]
    if duration is not None:
        end = min(float(duration), end)
    if end <= start:
        return list(span)
    return [round(start, 6), round(end, 6)]


@dataclass(frozen=True)
class OffsetCalibration:
    start_offset_s: float
    end_offset_s: float
    asr_config_hash: str
    model: str
    revision: str
    variant: str
    source_records_sha256: str

    @classmethod
    def load(cls, path):
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        if payload.get("method") != "constant-boundary-offsets":
            raise ValueError("Unsupported boundary calibration method.")
        for name in ("start_offset_s", "end_offset_s"):
            value = payload[name]
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
                raise ValueError(f"Invalid calibration value for {name}.")
        for name in ("asr_config_hash", "model", "revision", "variant", "source_records_sha256"):
            if not isinstance(payload[name], str) or not payload[name]:
                raise ValueError(f"Missing calibration provenance: {name}.")
        return cls(**{name: payload[name] for name in cls.__dataclass_fields__})

    def validate_context(self, *, asr_config_hash, model, revision, variant):
        actual = (asr_config_hash, model, revision, variant)
        expected = (self.asr_config_hash, self.model, self.revision, self.variant)
        if actual != expected:
            raise ValueError(
                f"Boundary calibration does not match this pipeline: {actual!r} != {expected!r}"
            )

    def apply(
        self, span: Sequence[float], duration: Optional[float] = None
    ) -> Tuple[float, float]:
        proposed = adjusted_span(span, (self.start_offset_s, self.end_offset_s), duration)
        return proposed[0], proposed[1]

    def metadata(self):
        return {"method": "constant-boundary-offsets", **asdict(self)}
