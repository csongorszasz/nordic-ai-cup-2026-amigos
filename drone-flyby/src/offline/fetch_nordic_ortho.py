"""Fetch Danish orthophoto strips for synthetic background data.

Klimadatastyrelsen / Dataforsyningen serves the national spring orthophoto as
open data. The WMS product ``orto_foraar_DAF`` carries the native-resolution
layers (``orto_foraar_12_5`` = 12.5 cm, ``orto_foraar_10`` = 10 cm,
``orto_foraar_cir`` = colour IR) in true-meter CRS such as EPSG:25832, which
makes the ground geometry exact - unlike a Web-Mercator cache, where ground
scale depends on latitude.

The challenge's simulated camera samples the ground at 0.2395 m/px (13.89 m of
forward flight between 58 source pixels). This fetcher therefore:

1. converts a start lat/lon to UTM 32N (pure-Python Snyder series, no pyproj);
2. computes the strip rectangle: ``length_m`` along the flight axis,
   ``width_m`` across it;
3. walks the *target* pixel grid in blocks, mapping each block back to ground
   meters so seams are impossible - every target pixel is written exactly once;
4. requests each block at the native layer resolution (WMS GetMap, EPSG:25832);
5. downsamples the block to the target GSD with INTER_AREA and pastes it.

Output (paths relative to the repository root):

    data/nordic_ortho/denmark/<site>/strip.png    BGR image, north-up
    data/nordic_ortho/denmark/<site>/strip.json   provenance + geometry

The access token is read from the ``DATAFORSYNINGEN_TOKEN`` environment
variable (free account at dataforsyningen.dk); it is never written to disk or
embedded here. Data license: open data, attribution
"Dataforsyningen / Klimadatastyrelsen" is required on derived datasets.

    DATAFORSYNINGEN_TOKEN=... python src/offline/fetch_nordic_ortho.py \
        --site billund_airport --length-m 4000 --width-m 1000
"""

from __future__ import annotations

import argparse
import json
import math
import os
import time
import urllib.parse
import urllib.request
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Tuple

import cv2
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]

WMS_URL = "https://api.dataforsyningen.dk/orto_foraar_DAF"
DEFAULT_LAYER = "orto_foraar_12_5"  # 12.5 cm native; orto_foraar_10 is sharper
CRS = "EPSG:25832"  # UTM 32N, true meters over Denmark
TOKEN_ENV = "DATAFORSYNINGEN_TOKEN"

TARGET_GSD = 0.2395  # challenge camera GSD (13.8889 m / 58 px), m/px
NATIVE_GSD = 0.125  # orto_foraar_12_5 native resolution, m/px

LICENSE = "Open data - attribution: Dataforsyningen / Klimadatastyrelsen"

# Preset sites chosen for challenge-relevant content: airfields carry hangars,
# jets and towers; harbours carry industrial clutter and towers. Those are
# context-rich but contain *real* lookalikes of the challenge classes, so use
# them only after manual review or for cropped sub-regions.
#
# The residential presets are deliberately lookalike-free backgrounds (dense
# housing + streets, no airfields, harbours, masts, turbines or military
# objects) - safe to composite sprites onto without label conflicts.
SITES: Dict[str, Tuple[float, float]] = {
    # Airports / harbours (contain real challenge-object lookalikes)
    "billund_airport": (55.7403, 9.1518),
    "aalborg_airport": (57.0928, 9.8494),
    "kastrup_airport": (55.6180, 12.6560),
    "aarhus_harbor": (56.1500, 10.2200),
    "odense_industrial": (55.4038, 10.4024),
    "esbjerg_harbor": (55.4676, 8.4520),
    # Residential / conflict-free backgrounds
    "ishoej_residential": (55.6157, 12.3523),
    "hoeje_taastrup": (55.6540, 12.3140),
    "birkerod_residential": (55.8436, 12.4300),
    "viby_aarhus": (56.1166, 10.1790),
    "hjallese_odense": (55.3850, 10.4160),
    "seest_kolding": (55.4780, 9.4340),
    "silkeborg_north": (56.1860, 9.5480),
    "hjoerring_east": (57.4640, 9.9950),
    "holbaek_east": (55.7220, 11.7700),
}

# --------------------------------------------------------------------------- #
# WGS84 -> UTM 32N (Snyder series; verified in tests against a numerical
# meridian-arc integration, so no pyproj dependency is needed)
# --------------------------------------------------------------------------- #

_A = 6378137.0
_F = 1.0 / 298.257223563
_E2 = _F * (2.0 - _F)
_E4 = _E2 * _E2
_E6 = _E4 * _E2
_EP2 = _E2 / (1.0 - _E2)
_K0 = 0.9996
_LON0 = math.radians(9.0)  # central meridian of zone 32N


def latlon_to_utm32n(lat_deg: float, lon_deg: float) -> Tuple[float, float]:
    """Convert WGS84 lat/lon (degrees) to UTM 32N easting/northing (meters)."""
    phi = math.radians(lat_deg)
    lam = math.radians(lon_deg)

    sin_phi = math.sin(phi)
    cos_phi = math.cos(phi)
    tan_phi = math.tan(phi)

    n = _A / math.sqrt(1.0 - _E2 * sin_phi * sin_phi)
    t = tan_phi * tan_phi
    c = _EP2 * cos_phi * cos_phi
    a = (lam - _LON0) * cos_phi

    m = _A * (
        (1.0 - _E2 / 4.0 - 3.0 * _E4 / 64.0 - 5.0 * _E6 / 256.0) * phi
        - (3.0 * _E2 / 8.0 + 3.0 * _E4 / 32.0 + 45.0 * _E6 / 1024.0) * math.sin(2.0 * phi)
        + (15.0 * _E4 / 256.0 + 45.0 * _E6 / 1024.0) * math.sin(4.0 * phi)
        - (35.0 * _E6 / 3072.0) * math.sin(6.0 * phi)
    )

    easting = _K0 * n * (
        a
        + (1.0 - t + c) * a**3 / 6.0
        + (5.0 - 18.0 * t + t * t + 72.0 * c - 58.0 * _EP2) * a**5 / 120.0
    ) + 500000.0
    northing = _K0 * (
        m
        + n * tan_phi * (
            a * a / 2.0
            + (5.0 - t + 9.0 * c + 4.0 * c * c) * a**4 / 24.0
            + (61.0 - 58.0 * t + t * t + 600.0 * c - 330.0 * _EP2) * a**6 / 720.0
        )
    )
    return easting, northing


# --------------------------------------------------------------------------- #
# Fetching
# --------------------------------------------------------------------------- #

def _http_get_bytes(url: str, timeout: float = 60.0) -> bytes:
    request = urllib.request.Request(url, headers={"User-Agent": "drone-flyby-synthetic-data/1.0"})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return response.read()


def iter_target_blocks(
    width: int, height: int, block_px: int
) -> Iterator[Tuple[int, int, int, int]]:
    """Cover a W x H target grid with (x0, y0, w, h) blocks; last ones clip."""
    for y0 in range(0, height, block_px):
        for x0 in range(0, width, block_px):
            yield x0, y0, min(block_px, width - x0), min(block_px, height - y0)


@dataclass
class StripSpec:
    """A straight corridor of background imagery, axis-aligned in UTM 32N."""

    site: str
    center_e: float
    center_n: float
    length_m: float  # along the flight axis
    width_m: float  # across it
    direction: str  # "ew", "we", "ns" or "sn"
    target_gsd: float = TARGET_GSD
    native_gsd: float = NATIVE_GSD
    layer: str = DEFAULT_LAYER

    def bbox(self) -> Tuple[float, float, float, float]:
        """(e0, n0, e1, n1) with the length axis per ``direction``."""
        along = self.length_m / 2.0
        across = self.width_m / 2.0
        if self.direction in ("ew", "we"):
            e0, e1 = self.center_e - along, self.center_e + along
            n0, n1 = self.center_n - across, self.center_n + across
        else:
            e0, e1 = self.center_e - across, self.center_e + across
            n0, n1 = self.center_n - along, self.center_n + along
        return e0, n0, e1, n1

    def target_size(self) -> Tuple[int, int]:
        """(width_px, height_px) of the north-up target image."""
        e0, n0, e1, n1 = self.bbox()
        return round((e1 - e0) / self.target_gsd), round((n1 - n0) / self.target_gsd)


def _request_block(
    spec: StripSpec,
    block: Tuple[int, int, int, int],
    token: str,
    timeout: float,
    attempts: int = 3,
) -> np.ndarray:
    """Fetch one target block: native GSD from the WMS, downsampled to target.

    The strip is north-up: target row ``y0`` is anchored to the *maximum*
    northing, so the block's ground window runs from
    ``n1 - (y0 + h_t) * gsd`` (south edge) up to ``n1 - y0 * gsd`` (north
    edge). WMS GetMap returns ``maxy`` on the top image row, which then pastes
    straight onto strip row ``y0``.
    """
    x0, y0, w_t, h_t = block
    e0, _, _, n1 = spec.bbox()
    # Ground extent of this block, exact in target pixels.
    bx0 = e0 + x0 * spec.target_gsd
    bx1 = bx0 + w_t * spec.target_gsd
    by1 = n1 - y0 * spec.target_gsd
    by0 = by1 - h_t * spec.target_gsd

    n_w = max(1, round(w_t * spec.target_gsd / spec.native_gsd))
    n_h = max(1, round(h_t * spec.target_gsd / spec.native_gsd))
    if max(n_w, n_h) > 10000:
        raise ValueError(f"Block too large for the WMS: {n_w}x{n_h} px")

    params = {
        "service": "WMS",
        "version": "1.1.1",
        "request": "GetMap",
        "layers": spec.layer,
        "styles": "",
        "format": "image/png",
        "srs": CRS,
        "bbox": f"{bx0:.3f},{by0:.3f},{bx1:.3f},{by1:.3f}",
        "width": str(n_w),
        "height": str(n_h),
        "token": token,
    }
    url = f"{WMS_URL}?{urllib.parse.urlencode(params)}"

    last_error: Optional[Exception] = None
    for attempt in range(attempts):
        try:
            payload = _http_get_bytes(url, timeout=timeout)
            image = cv2.imdecode(np.frombuffer(payload, dtype=np.uint8), cv2.IMREAD_COLOR)
            if image is None:
                raise ValueError("WMS returned an undecodable image")
            if image.shape[:2] != (n_h, n_w):
                image = cv2.resize(image, (n_w, n_h), interpolation=cv2.INTER_AREA)
            break
        except Exception as exc:  # noqa: BLE001 - retry any transport failure
            last_error = exc
            if attempt + 1 < attempts:
                time.sleep(1.0 * (attempt + 1))
    else:
        raise RuntimeError(f"WMS GetMap failed after {attempts} attempts: {last_error}")

    return cv2.resize(image, (w_t, h_t), interpolation=cv2.INTER_AREA)


def fetch_strip(
    spec: StripSpec,
    token: str,
    output_dir: Path,
    block_px: int = 2048,
    request_delay_s: float = 0.2,
) -> Path:
    """Download the whole strip and write strip.png + strip.json."""
    width, height = spec.target_size()
    strip = np.zeros((height, width, 3), dtype=np.uint8)

    blocks = list(iter_target_blocks(width, height, block_px))
    print(
        f"Fetching {spec.site}: {width}x{height} px target "
        f"({spec.length_m:.0f}x{spec.width_m:.0f} m at {spec.target_gsd} m/px), "
        f"{len(blocks)} WMS requests"
    )
    for index, block in enumerate(blocks, start=1):
        chunk = _request_block(spec, block, token, timeout=60.0)
        x0, y0, w_t, h_t = block
        strip[y0 : y0 + h_t, x0 : x0 + w_t] = chunk
        print(f"  [{index}/{len(blocks)}] block {block} done", flush=True)
        time.sleep(request_delay_s)

    output_dir.mkdir(parents=True, exist_ok=True)
    image_path = output_dir / "strip.png"
    cv2.imwrite(str(image_path), strip, [cv2.IMWRITE_PNG_COMPRESSION, 3])

    e0, n0, e1, n1 = spec.bbox()
    metadata = {
        "site": spec.site,
        "provider": "Dataforsyningen / Klimadatastyrelsen",
        "service": WMS_URL,
        "layer": spec.layer,
        "crs": CRS,
        "license": LICENSE,
        "native_gsd_m": spec.native_gsd,
        "target_gsd_m": spec.target_gsd,
        "bbox_utm32": [round(v, 2) for v in (e0, n0, e1, n1)],
        "size_px": [width, height],
        "flight_direction": spec.direction,
        "north_up": True,
        "length_m": spec.length_m,
        "width_m": spec.width_m,
        "n_requests": len(blocks),
        "fetched_at": datetime.now(timezone.utc).isoformat(),
    }
    (output_dir / "strip.json").write_text(json.dumps(metadata, indent=2))
    print(f"Strip written to {image_path} ({image_path.stat().st_size / 1e6:.1f} MB)")
    return image_path


def fix_existing_strips(output_root: Path) -> int:
    """Flip pre-fix strips (south-up) in place to north-up.

    Blocks were fetched correctly; only the vertical assembly was mirrored, so
    a single ``flipud`` is an exact, lossless correction. Strips carrying
    ``"north_up": true`` are left untouched, making this idempotent.
    """
    fixed = 0
    for metadata_path in sorted(output_root.glob("*/strip.json")):
        metadata = json.loads(metadata_path.read_text())
        if metadata.get("north_up"):
            continue
        image_path = metadata_path.parent / "strip.png"
        strip = cv2.imread(str(image_path))
        if strip is None:
            print(f"  {metadata_path.parent.name}: strip.png unreadable, skipped")
            continue
        cv2.imwrite(str(image_path), np.ascontiguousarray(strip[::-1]),
                    [cv2.IMWRITE_PNG_COMPRESSION, 3])
        metadata["north_up"] = True
        metadata_path.write_text(json.dumps(metadata, indent=2))
        fixed += 1
        print(f"  {metadata_path.parent.name}: flipped to north-up")
    return fixed


def main() -> int:
    parser = argparse.ArgumentParser(description="Fetch a Danish orthophoto strip.")
    parser.add_argument("--site", choices=sorted(SITES), default=None,
                        help="Preset site; see SITES for coordinates.")
    parser.add_argument("--lat", type=float, default=None, help="Custom center latitude.")
    parser.add_argument("--lon", type=float, default=None, help="Custom center longitude.")
    parser.add_argument("--name", default=None, help="Output folder name (default: site).")
    parser.add_argument("--length-m", type=float, default=4000.0)
    parser.add_argument("--width-m", type=float, default=1000.0)
    parser.add_argument("--direction", choices=("ew", "we", "ns", "sn"), default="we")
    parser.add_argument("--target-gsd", type=float, default=TARGET_GSD)
    parser.add_argument("--native-gsd", type=float, default=NATIVE_GSD)
    parser.add_argument("--layer", default=DEFAULT_LAYER)
    parser.add_argument("--block-px", type=int, default=2048)
    parser.add_argument(
        "--fix-existing",
        action="store_true",
        help="Flip already-fetched south-up strips to north-up and exit.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=REPO_ROOT / "data" / "nordic_ortho" / "denmark",
    )
    arguments = parser.parse_args()

    if arguments.fix_existing:
        if not arguments.output_root.is_dir():
            print(f"No output root at {arguments.output_root}")
            return 1
        fixed = fix_existing_strips(arguments.output_root)
        print(f"Fixed {fixed} strip(s)")
        return 0

    token = os.getenv(TOKEN_ENV, "").strip()
    if not token:
        print(f"Set {TOKEN_ENV} (free account at dataforsyningen.dk)")
        return 1

    if arguments.site:
        lat, lon = SITES[arguments.site]
        name = arguments.name or arguments.site
    elif arguments.lat is not None and arguments.lon is not None:
        lat, lon = arguments.lat, arguments.lon
        name = arguments.name or f"custom_{lat:.4f}_{lon:.4f}"
    else:
        parser.error("Provide --site or both --lat and --lon")
        return 1

    center_e, center_n = latlon_to_utm32n(lat, lon)
    spec = StripSpec(
        site=name,
        center_e=center_e,
        center_n=center_n,
        length_m=arguments.length_m,
        width_m=arguments.width_m,
        direction=arguments.direction,
        target_gsd=arguments.target_gsd,
        native_gsd=arguments.native_gsd,
        layer=arguments.layer,
    )
    fetch_strip(spec, token, arguments.output_root / name, block_px=arguments.block_px)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
