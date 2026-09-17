"""Download aerial backgrounds for the synthetic dataset.

The validation scene is a city (Copenhagen) while the supplied training scene is
forest and lake, which is why the model calls roads "hangar". We need varied
backgrounds to paste sprites onto, and we must not train on the recorded
validation frames: they are our only honest test set.

Source: USGS NAIP (`imagery.nationalmap.gov`), public domain, no API key. Native
resolution is 0.6 m/px and we ask for 0.21 m/px, the scale of the challenge
imagery, so tiles come out slightly soft - the challenge's own 3D tiles are soft
too, and geometry matters more here than texture.

    python training/fetch_backgrounds.py                    # every area, 2 tiles each
    python training/fetch_backgrounds.py --per-area 4       # more of each
    python training/fetch_backgrounds.py --areas harbour_la forest_or

Writes backgrounds/naip/<area>_<n>.jpg (3840x2160) plus a manifest with the
bounding box of each tile, so a tile can be traced back to its ground location.
"""

import argparse
import json
import math
import random
import time
from pathlib import Path
from urllib.parse import urlencode
from urllib.request import urlopen

ROOT = Path(__file__).resolve().parents[1]
SERVICE = 'https://imagery.nationalmap.gov/arcgis/rest/services/USGSNAIPPlus/ImageServer/exportImage'

WIDTH, HEIGHT = 3840, 2160
METRES_PER_PIXEL = 0.21  # measured on the challenge imagery (~600 m altitude)

# Ground types that produce our false positives (roads, roofs, harbours, hardstanding)
# plus the empty ones the detector should stay quiet on.
AREAS = {
    'city_chicago':     (41.8790, -87.6300, 'dense downtown, grid streets, flat roofs'),
    'city_houston':     (29.7500, -95.3700, 'downtown and freeway interchanges'),
    'suburb_phoenix':   (33.4800, -112.0900, 'suburban housing, pools, wide roads'),
    'suburb_atlanta':   (33.8400, -84.3800, 'wooded suburbs and arterials'),
    'harbour_la':       (33.7420, -118.2600, 'container port, cranes, stacked containers'),
    'harbour_seattle':  (47.5800, -122.3400, 'docks, warehouses, rail'),
    'marina_florida':   (26.1200, -80.1100, 'marina, moored boats, waterfront'),
    'airport_dfw':      (32.8970, -97.0400, 'runways, aprons, parked aircraft'),
    'airfield_mojave':  (35.0590, -118.1520, 'desert airfield, stored airframes'),
    'industry_gary':    (41.6100, -87.3400, 'heavy industry, tanks, stockpiles'),
    'railyard_kc':      (39.1000, -94.5700, 'rail yard, rolling stock, sheds'),
    'warehouse_inland': (34.0400, -117.4000, 'distribution sheds, lorry parks'),
    'farm_iowa':        (42.0300, -93.6200, 'farmland, barns, silos'),
    'forest_oregon':    (44.0500, -122.3000, 'conifer forest and logging roads'),
    'lake_minnesota':   (46.8000, -94.6000, 'lakes, woodland, cabins'),
    'coast_maine':      (43.8000, -69.8000, 'rocky coast, small harbours'),
    'desert_nevada':    (36.6000, -115.2000, 'bare desert, tracks'),
    'quarry_utah':      (40.5200, -112.1500, 'quarry, spoil heaps, plant'),
}


def web_mercator(lon: float, lat: float):
    x = lon * 20037508.34 / 180.0
    y = math.log(math.tan((90 + lat) * math.pi / 360.0)) / (math.pi / 180.0) * 20037508.34 / 180.0
    return x, y


def tile_bbox(lat: float, lon: float, jitter_m: float, rng: random.Random):
    """Web-Mercator bbox for one 3840x2160 tile at METRES_PER_PIXEL of ground."""
    # Mercator metres are stretched by 1/cos(latitude); correct so the ground scale is right.
    stretch = 1.0 / math.cos(math.radians(lat))
    half_w = WIDTH * METRES_PER_PIXEL * stretch / 2
    half_h = HEIGHT * METRES_PER_PIXEL * stretch / 2
    cx, cy = web_mercator(lon, lat)
    cx += rng.uniform(-jitter_m, jitter_m) * stretch
    cy += rng.uniform(-jitter_m, jitter_m) * stretch
    return cx - half_w, cy - half_h, cx + half_w, cy + half_h


def fetch(bbox, timeout: int) -> bytes:
    query = urlencode({
        'bbox': ','.join(f'{v:.2f}' for v in bbox),
        'bboxSR': 3857,
        'imageSR': 3857,
        'size': f'{WIDTH},{HEIGHT}',
        'format': 'jpg',
        'f': 'image',
    })
    with urlopen(f'{SERVICE}?{query}', timeout=timeout) as response:
        return response.read()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', default=str(ROOT / 'backgrounds' / 'naip'))
    parser.add_argument('--areas', nargs='*', help='area keys to fetch (default: all)')
    parser.add_argument('--per-area', type=int, default=2)
    parser.add_argument('--jitter', type=float, default=1500.0, help='metres of random offset per tile')
    parser.add_argument('--timeout', type=int, default=120)
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    manifest_path = out / 'manifest.json'
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}

    rng = random.Random(args.seed)
    keys = args.areas or sorted(AREAS)
    for key in keys:
        if key not in AREAS:
            raise SystemExit(f'unknown area {key!r}; known: {", ".join(sorted(AREAS))}')
        lat, lon, description = AREAS[key]
        for n in range(args.per_area):
            name = f'{key}_{n}.jpg'
            path = out / name
            if path.exists():
                print(f'{name}: already there')
                continue
            bbox = tile_bbox(lat, lon, args.jitter if n else 0.0, rng)
            for attempt in range(3):
                try:
                    data = fetch(bbox, args.timeout)
                    break
                except Exception as exc:  # network hiccups are common on a home line
                    print(f'{name}: attempt {attempt + 1} failed ({exc})')
                    time.sleep(3)
            else:
                print(f'{name}: giving up')
                continue
            path.write_bytes(data)
            manifest[name] = {'area': key, 'description': description, 'bbox_3857': bbox,
                              'metres_per_pixel': METRES_PER_PIXEL, 'source': 'USGS NAIP (public domain)'}
            manifest_path.write_text(json.dumps(manifest, indent=1, sort_keys=True))
            print(f'{name}: {len(data) / 1e6:.1f} MB  {description}')

    print(f'{len(list(out.glob("*.jpg")))} backgrounds in {out}')


if __name__ == '__main__':
    main()
