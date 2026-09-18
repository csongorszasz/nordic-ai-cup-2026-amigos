"""Download Helsinki 3D reality-mesh tiles for render_city.py, keeping only what it needs.

Each 2 km tile is a 0.1-3.5 GB zip of OBJ quadtree levels; render_city.py only reads
level 19, about a third of it. This downloads one zip at a time, extracts L19 and the
metadata, and deletes the zip, so disk use peaks at one zip. Finished tiles are
skipped, so an interrupted run resumes.

Tile names are <N km><E km> in GK25 (EPSG:3879); the whole city is 122 tiles, 190 GB.
The default set is picked for variety (airfield, industry, forest, fields, rail yard,
suburbs, city centre, harbours). Data: City of Helsinki, "Helsinki 3D" reality mesh
2017, CC BY 4.0.

    python training/fetch_city_mesh.py                  # the default set
    python training/fetch_city_mesh.py 682502 680502    # specific tiles
"""

import argparse
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'backgrounds' / 'helsinki3d'
BASE = 'http://3d.hel.ninja/data/mesh/Helsinki3D-MESH_2017_OBJ_2km-250m_ZIP/'
LEVEL = 19

DEFAULT_TILES = {
    '676510': 'Vuosaari harbour',
    '670494': 'Jatkasaari west harbour',
    '682502': 'Malmi airport',
    '680502': 'Kivikko industrial',
    '678494': 'Central Park forest',
    '678500': 'Viikki fields',
    '676496': 'Pasila rail yard',
    '684502': 'Suurmetsa suburbs',
    '672496': 'Kallio / city centre',
    '686510': 'north-east fields and forest',
}


def salvage(archive: Path, target: Path, tile: str):
    """Pull what is readable out of a zip whose end is corrupt on the server (684502 is).

    Without its central directory zipfile cannot open it, but bsdtar reading it as a stream
    walks the local headers in order. The last subtile written may be cut, so it is dropped;
    metadata.xml (lost with the tail) is the same for every tile and is copied from another.
    """
    donor = next((d / 'metadata.xml' for d in sorted(OUT.glob('tile_*')) if (d / 'metadata.xml').exists()), None)
    if shutil.which('bsdtar') is None or donor is None:
        print(f'{tile}: corrupt on the server and cannot be salvaged here (needs bsdtar); skipped', flush=True)
        archive.unlink()
        return None
    print(f'{tile}: corrupt on the server; salvaging what precedes the damage', flush=True)
    target.mkdir(parents=True, exist_ok=True)
    with open(archive, 'rb') as stream:
        subprocess.run(['bsdtar', '-xf', '-', '-C', str(target), '--include', f'*_L{LEVEL}_*'], stdin=stream)
    files = [f for f in target.rglob('*') if f.is_file()]
    if files:
        shutil.rmtree(max(files, key=lambda f: f.stat().st_mtime).parent)
    shutil.copy(donor, target / 'metadata.xml')
    return [f for f in target.rglob('*') if f.is_file()]


def fetch(tile: str):
    target = OUT / f'tile_{tile}'
    if (target / 'metadata.xml').exists():
        print(f'{tile}: already there')
        return
    archive = OUT / f'Helsinki3D_2017_OBJ_{tile}x2.zip'
    url = f'{BASE}{archive.name}'
    print(f'{tile} ({DEFAULT_TILES.get(tile, "custom")}): downloading {url}', flush=True)
    # curl resumes a partial file (-C -); urllib would start over after an interruption.
    subprocess.run(['curl', '-sSL', '--fail', '-C', '-', '--retry', '5', '-o', str(archive), url], check=True)
    if not zipfile.is_zipfile(archive):
        # Full size but broken, e.g. a reboot lost the unwritten tail: resuming cannot fix it.
        print(f'{tile}: broken zip, downloading again', flush=True)
        archive.unlink()
        subprocess.run(['curl', '-sSL', '--fail', '--retry', '5', '-o', str(archive), url], check=True)
    if zipfile.is_zipfile(archive):
        with zipfile.ZipFile(archive) as z:
            names = [n for n in z.namelist() if f'_L{LEVEL}_' in n or n.endswith('metadata.xml')]
            z.extractall(target, members=names)
    else:
        names = salvage(archive, target, tile)
        if names is None:
            return
    archive.unlink()
    size = sum(f.stat().st_size for f in target.rglob('*') if f.is_file())
    print(f'{tile}: kept {len(names)} files, {size / 2**30:.2f} GB; {shutil.disk_usage(OUT).free / 2**30:.0f} GB free', flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('tiles', nargs='*', default=list(DEFAULT_TILES))
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    for tile in args.tiles:
        if shutil.disk_usage(OUT).free < 8 * 2**30:
            sys.exit('less than 8 GB free; stopping')
        fetch(tile)


if __name__ == '__main__':
    main()
