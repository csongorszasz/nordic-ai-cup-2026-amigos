"""Convert manually downloaded NLS (Finland) orthophoto sheets to usable images.

Read embedded GML georeferencing, including metadata after the image codestream.
Neither the directory name nor a presumed sheet size establishes pixel scale.
Files without usable metadata require an explicit, recorded GSD override.

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
import math
import re
import struct
import sys
import xml.etree.ElementTree as ET
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

Image.MAX_IMAGE_PIXELS = 150_000_000

SHEET_RE = re.compile(r"([A-Z]\d{2})(\d)(\d)([A-H])$")
LICENSE = "CC-BY 4.0 - Maanmittauslaitos (National Land Survey of Finland)"
CRS = "ETRS-TM35FIN (EPSG:3067)"


def read_georeference(path: Path) -> Optional[Dict]:
    """Read bounded JP2 XML boxes without decoding or scanning the codestream."""
    def xml_boxes(handle, end, depth=0):
        if depth > 8:
            raise ValueError("Excessively nested JPEG-2000 metadata")
        while handle.tell() < end:
            start = handle.tell()
            header = handle.read(8)
            if len(header) != 8:
                raise ValueError("Truncated JPEG-2000 box header")
            length, kind = struct.unpack(">I4s", header)
            header_size = 8
            if length == 1:
                extended = handle.read(8)
                if len(extended) != 8:
                    raise ValueError("Truncated extended JPEG-2000 box")
                length = struct.unpack(">Q", extended)[0]
                header_size = 16
            elif length == 0:
                length = end - start
            if length < header_size or start + length > end:
                raise ValueError("Invalid JPEG-2000 box length")
            if kind in (b"asoc", b"jp2h"):
                yield from xml_boxes(handle, start + length, depth + 1)
            elif kind == b"xml ":
                if length - header_size > 4 * 1024 * 1024:
                    raise ValueError("JPEG-2000 XML metadata exceeds the bounded reader limit")
                yield handle.read(length - header_size)
            handle.seek(start + length)

    with Path(path).open("rb") as handle:
        for data in xml_boxes(handle, Path(path).stat().st_size):
            # NLS GML boxes may include a trailing C-string terminator.
            root = ET.fromstring(data.rstrip(b"\x00"))
            for grid in root.iter():
                if grid.tag.rsplit("}", 1)[-1] != "RectifiedGrid":
                    continue
                vectors, origin, crs_names = [], None, []
                for element in grid.iter():
                    name = element.tag.rsplit("}", 1)[-1]
                    if element.get("srsName"):
                        crs_names.append(element.get("srsName"))
                    if name == "offsetVector":
                        vectors.append([float(value) for value in (element.text or "").split()])
                    elif name == "pos":
                        origin = [float(value) for value in (element.text or "").split()]
                if len(vectors) != 2 or any(len(vector) != 2 for vector in vectors):
                    continue
                if not crs_names or any(not name.endswith(("::3067", "/3067", ":3067")) for name in crs_names):
                    raise ValueError("Embedded grid is not a verified EPSG:3067 metre grid")
                first, second = vectors
                gsd_x, gsd_y = math.hypot(*first), math.hypot(*second)
                dot = sum(a * b for a, b in zip(first, second))
                if (not all(math.isfinite(value) for vector in vectors for value in vector)
                        or gsd_x <= 0 or gsd_y <= 0
                        or not math.isclose(gsd_x, gsd_y, rel_tol=1e-6)
                        or abs(dot) > gsd_x * gsd_y * 1e-6):
                    raise ValueError("Embedded grid must have finite, orthogonal square pixels")
                if origin is not None and (len(origin) != 2 or not all(math.isfinite(v) for v in origin)):
                    raise ValueError("Invalid embedded grid origin")
                return {
                    "gsd_m": gsd_x, "gsd_source": "embedded-gml", "crs": CRS,
                    "offset_vectors": vectors, "origin_pixel_center": origin,
                }
    return None


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


def convert_sheet(sheet: SheetInfo, output_dir: Path, fmt: str, quality: int,
                  gsd_override: Optional[float] = None) -> Dict:
    """Decode one JP2 sheet and write it as a plain image plus metadata."""
    georeference = read_georeference(sheet.path)
    if gsd_override is not None and (not math.isfinite(gsd_override) or gsd_override <= 0):
        raise ValueError("GSD override must be a positive finite number of metres")
    if georeference is not None:
        if gsd_override is not None and not math.isclose(gsd_override, georeference["gsd_m"], rel_tol=1e-6):
            raise ValueError("GSD override contradicts embedded georeferencing")
    elif gsd_override is not None:
        georeference = {"gsd_m": gsd_override, "gsd_source": "explicit-override", "crs": None}
    else:
        raise ValueError(f"No usable embedded georeference in {sheet.path}; provide --gsd-m explicitly")
    out_dir = output_dir / sheet.year
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{sheet.sheet_id}{sheet.letter}.{fmt}"
    skipped = out_path.exists()

    with Image.open(sheet.path) as image:
        width, height = image.size
        if width * height > 150_000_000:
            raise ValueError("Orthophoto exceeds the supported 150-megapixel working set")
        if not skipped:
            if image.mode != "RGB":
                image = image.convert("RGB")
            if fmt == "png":
                image.save(out_path, format="PNG", optimize=False)
            else:
                image.save(out_path, format="JPEG", quality=quality, optimize=True)

    return {
        "sheet": sheet.sheet_id + sheet.letter,
        "skipped": skipped,
        "path": str(out_path),
        "year": sheet.year,
        "source": str(sheet.path),
        "width_px": width,
        "height_px": height,
        **georeference,
        "tile_size_m": width * georeference["gsd_m"],
        "tile_height_m": height * georeference["gsd_m"],
        "license": LICENSE,
        "bytes_out": out_path.stat().st_size,
    }


def build_manifest(entries: List[Dict]) -> Dict:
    converted = [entry for entry in entries if "width_px" in entry]
    sizes = {e["width_px"] for e in converted} | {e["height_px"] for e in converted}
    gsds = sorted({entry["gsd_m"] for entry in converted})
    crs_values = {entry.get("crs") for entry in converted}
    return {
        "provider": "Maanmittauslaitos / National Land Survey of Finland",
        "service": "Tiedostopalvelu (File service REST), product mara_v_25000_50",
        "license": LICENSE,
        "crs": next(iter(crs_values)) if len(crs_values) == 1 else None,
        "gsd_m": gsds[0] if len(gsds) == 1 else None,
        "gsd_values_m": gsds,
        "gsd_source": "per-tile metadata",
        "gsd_note": "Embedded GML is authoritative; explicit overrides are recorded per tile.",
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
    parser.add_argument("--gsd-m", type=float, help="Explicit fallback only for files without embedded GML.")
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
        entry = convert_sheet(sheet, arguments.output_dir, arguments.format, arguments.quality, arguments.gsd_m)
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
