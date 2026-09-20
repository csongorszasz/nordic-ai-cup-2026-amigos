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
    # Added after counting the live model's false positives on Copenhagen: over 249 frames it
    # reported 1255 jammers, 439 spacecraft and 177 condors in a scene that has none of them,
    # firing on moored boats, field sheds, parked lorries, car yards and bushes. These areas are
    # those same things, so the detector meets them with nothing to find.
    'marina_annapolis': (38.9500, -76.4800, 'moored yachts, pontoons, boatyard'),
    'marina_lakeunion': (47.6500, -122.3300, 'houseboats and marina berths in a city'),
    'marina_michigan':  (42.1000, -86.4800, 'lake marina, dinghies, slipways'),
    'boatyard_newport': (41.4900, -71.3200, 'hauled-out hulls on hardstanding'),
    'allotments_pdx':   (45.5300, -122.6500, 'allotment plots, sheds, polytunnels'),
    'trailerpark_tampa': (27.9500, -82.3500, 'rows of caravans and mobile homes'),
    'dealership_dallas': (32.9000, -96.8000, 'ranked new cars and lorries'),
    'carpark_ohio':     (39.9800, -83.1300, 'mall car parks, service yards'),
    'cemetery_chicago': (41.8600, -87.8300, 'rows of small pale rectangles on grass'),
    'greenhouse_salinas': (36.6800, -121.6500, 'glasshouse blocks and packing sheds'),
    'solarfarm_nevada': (35.8000, -115.4700, 'panel arrays, inverter cabins'),
    'windfarm_texas':   (32.3000, -100.9000, 'turbines, access tracks, pads'),
    'golf_scottsdale':  (33.6000, -111.9200, 'fairways, bunkers, cart sheds'),
    'sportsfields_denver': (39.7500, -104.9000, 'pitches, dugouts, floodlight masts'),
    'construction_austin': (30.3500, -97.7200, 'plant, spoil, part-built slabs'),
    'orchard_yakima':   (46.5800, -120.5000, 'tree rows, hail netting, bins'),
    'pasture_wisconsin': (43.6000, -89.7000, 'round bales and stock troughs on grass'),
    'scrub_newmexico':  (35.1000, -106.5000, 'scattered bushes on bare ground'),
    'marsh_louisiana':  (29.7000, -90.1000, 'water channels, spoil banks, jetties'),
    'coast_oregon':     (44.6300, -124.0500, 'rocky shore, small craft, jetties'),
}

# HELD-OUT areas: never fetched before the test set was built, so no training run has seen
# them. training/holdout_scene.py pastes objects on these to score background generalisation
# with exact labels. Keep them out of AREAS so a training fetch cannot pull them in.
HOLDOUT_AREAS = {
    'ho_valley_ca':     (36.3300, -119.6500, 'irrigated valley floor, packing sheds'),
    'ho_coast_nc':      (34.7200, -76.7300, 'barrier coast, creeks, small docks'),
    'ho_plains_ks':     (38.5000, -98.2000, 'section-line farmland, grain bins'),
    'ho_city_denver':   (39.7400, -105.0000, 'downtown grid, flat roofs, lots'),
    'ho_hills_tn':      (35.9000, -84.1500, 'wooded hills, ridge roads'),
    'ho_port_savannah': (32.1300, -81.1400, 'container terminal, barges'),
    'ho_airbase_tucson': (32.1700, -110.8800, 'apron, hardstanding, stored airframes'),
    'ho_lakes_mn':      (46.3000, -94.2000, 'lakes, timber, cabins'),
    'ho_suburb_tx':     (30.1800, -95.4500, 'new suburbs, cul-de-sacs, ponds'),
    'ho_desert_ut':     (38.5700, -109.5500, 'red rock, wash channels, tracks'),
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
    parser.add_argument('--holdout', action='store_true',
                        help='fetch HOLDOUT_AREAS instead: backgrounds kept out of every training run')
    args = parser.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    manifest_path = out / 'manifest.json'
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}

    rng = random.Random(args.seed)
    pool = dict(AREAS, **HOLDOUT_AREAS) if args.holdout else AREAS
    keys = args.areas or sorted(HOLDOUT_AREAS if args.holdout else AREAS)
    for key in keys:
        if key not in pool:
            raise SystemExit(f'unknown area {key!r}; known: {", ".join(sorted(pool))}')
        lat, lon, description = pool[key]
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
