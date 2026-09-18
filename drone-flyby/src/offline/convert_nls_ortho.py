"""Convert manually downloaded NLS (Finland) orthophoto sheets to usable images.

The NLS File service delivers orthophotos as JPEG-2000 map sheets under
``mara_v_25000_50`` (the 1:25000 sheet grid). Each sheet, e.g. ``L3413``, is
subdivided into eight 3 km x 3 km tiles lettered ``A``-``H``, and each tile is
stored as a 12000 x 12000 px JPEG-2000 with **no embedded georeference**.

From the sheet grid this fixes the ground sample distance:

    3000 m / 12000 px = 0.25 m/px

which is marked ``"gsd_source": "inferred"`` in the manifest: it follows from
the NLS sheet geometry, not from file metadata, and should be re-verified
before any metric use.

Output layout (all paths relative to the repository root):

    data/nordic_ortho/finland/<year>/<sheet_id>.jpg   converted imagery
    data/nordic_ortho/finland/manifest.json           per-tile metadata

JPEG quality 95 is the default: these images are *backgrounds* for synthetic
scene rendering, and 10x smaller files keep the dataset manageable. Use
``--format png`` when lossless output is wanted.

The source tree under ``data/finland`` is left untouched.

    python src/offline/convert_nls_ortho.py
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]

try:  # Pillow wheels bundle OpenJPEG, which is what reads the NLS JP2 files.
    from PIL import Image
except Exception as exc:  # pragma: no cover - exercised only without Pillow
    raise SystemExit(f"Pillow is required to read JPEG-2000: {exc}") from exc

Image.MAX_IMAGE_PIXELS = None  # NLS sheets are 144 Mpx by design.

SHEET_RE = re.compile(r"([A-Z]\d{2})(\d)(\d)([A-H])$")
INFERRED_GSD = 0.25
INFERRED_TILE_METERS = 3000.0
LICENSE = "CC-BY 4.0 - Maanmittauslaitos (National Land Survey of Finland)"
CRS = "ETRS-TM35FIN (EPSG:3067)"


@dataclass(slots=True)
class SheetInfo:
    """One NLS map sheet tile parsed from its File-service path."""

    path: Path
    year: str
    sheet50k: str
    sheet_id: str
    letter: str

    @classmethod
    def from_path(cls, path: Path) -> Optional["SheetInfo"]:
        parts = path.parts
        if len(parts) < 5:
            return None
        year, sheet50k = parts[-5], parts[-4]
        match = SHEET_RE.search(path.stem)
        if match is None:
            return None
        return cls(
            path=path,
            year=year,
            sheet50k=sheet50k,
            sheet_id=match.group(1) + match.group(2) + match.group(3),
            letter=match.group(4),
        )


def find_jp2_sheets(input_dir: Path) -> List[SheetInfo]:
    """Every parseable JPEG-2000 sheet under the NLS File-service tree."""
    sheets = []
    for path in sorted(input_dir.rglob("*.jp2")):
        info = SheetInfo.from_path(path)
        if info is not None:
            sheets.append(info)
    return sheets


def convert_sheet(sheet: SheetInfo, output_dir: Path, fmt: str, quality: int) -> Dict:
    """Decode one JP2 sheet and write it as a plain image plus metadata."""
    out_dir = output_dir / sheet.year
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{sheet.sheet_id}{sheet.letter}.{fmt}"
    if out_path.exists():
        return {"sheet": sheet.sheet_id + sheet.letter, "skipped": True, "path": str(out_path)}

    with Image.open(sheet.path) as image:
        if image.mode != "RGB":
            image = image.convert("RGB")
        width, height = image.size
        if fmt == "png":
            image.save(out_path, format="PNG", optimize=False)
        else:
            image.save(out_path, format="JPEG", quality=quality, optimize=True)

    return {
        "sheet": sheet.sheet_id + sheet.letter,
        "skipped": False,
        "path": str(out_path),
        "year": sheet.year,
        "source": str(sheet.path),
        "width_px": width,
        "height_px": height,
        "gsd_m": INFERRED_GSD,
        "tile_size_m": width * INFERRED_GSD,
        "crs": CRS,
        "gsd_source": "inferred",
        "license": LICENSE,
        "bytes_out": out_path.stat().st_size,
    }


def build_manifest(entries: List[Dict]) -> Dict:
    converted = [e for e in entries if not e.get("skipped")]
    sizes = {e["width_px"] for e in converted} | {e["height_px"] for e in converted}
    return {
        "provider": "Maanmittauslaitos / National Land Survey of Finland",
        "service": "Tiedostopalvelu (File service REST), product mara_v_25000_50",
        "license": LICENSE,
        "crs": CRS,
        "gsd_m": INFERRED_GSD,
        "gsd_source": "inferred",
        "gsd_note": (
            "No georeference is embedded in the JP2 files. The GSD follows from "
            "the NLS sheet grid: each mara_v_25000_50 sheet splits into eight "
            "3 km x 3 km tiles stored as 12000 x 12000 px."
        ),
        "pixel_sizes": sorted(sizes),
        "years": sorted({e["year"] for e in converted}),
        "tiles": [e for e in entries if e.get("path")],
        "converted_at": datetime.now(timezone.utc).isoformat(),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Convert NLS JPEG-2000 ortho sheets to usable images.")
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=REPO_ROOT / "data" / "finland",
        help="Root of the NLS File-service download tree.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "data" / "nordic_ortho" / "finland",
        help="Where converted images and the manifest are written.",
    )
    parser.add_argument("--format", choices=("jpg", "png"), default="jpg")
    parser.add_argument("--quality", type=int, default=95)
    arguments = parser.parse_args()

    if not arguments.input_dir.is_dir():
        print(f"Input directory not found: {arguments.input_dir}")
        return 1

    sheets = find_jp2_sheets(arguments.input_dir)
    if not sheets:
        print(f"No JPEG-2000 sheets found under {arguments.input_dir}")
        return 1

    print(f"Converting {len(sheets)} sheets from {arguments.input_dir}")
    entries: List[Dict] = []
    for index, sheet in enumerate(sheets, start=1):
        entry = convert_sheet(sheet, arguments.output_dir, arguments.format, arguments.quality)
        status = "skipped (exists)" if entry.get("skipped") else (
            f"{entry['width_px']}x{entry['height_px']} -> {entry['bytes_out'] / 1e6:.1f} MB"
        )
        print(f"  [{index}/{len(sheets)}] {entry['sheet']} {status}", flush=True)
        entries.append(entry)

    manifest = build_manifest(entries)
    manifest_path = arguments.output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))
    print(f"Manifest written to {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
