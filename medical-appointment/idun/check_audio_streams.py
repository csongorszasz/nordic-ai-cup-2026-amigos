"""Verify in-memory MP3 decoding exactly matches the previous file input path."""

import io
import json
from pathlib import Path

import numpy as np
from faster_whisper.audio import decode_audio


def main():
    root = Path(__file__).resolve().parent.parent
    records = []
    paths = sorted((root / "data" / "audio").glob("*.mp3"))
    if len(paths) != 39:
        raise ValueError("The full supplied audio corpus is required.")
    for path in paths:
        expected = decode_audio(str(path), sampling_rate=16000)
        actual = decode_audio(io.BytesIO(path.read_bytes()), sampling_rate=16000)
        if expected.dtype != actual.dtype or not np.array_equal(expected, actual):
            raise ValueError(f"Stream decoding changed the waveform: {path.name}")
        records.append({
            "audio_filename": path.name, "samples": len(actual),
            "dtype": str(actual.dtype), "bit_identical": True,
        })
    output = root / "results" / "audio_stream_check.json"
    output.write_text(json.dumps({"conversations": len(records), "records": records}, indent=2))
    print(f"All {len(records)} MP3 waveforms are bit-identical through file and BytesIO decoding.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
