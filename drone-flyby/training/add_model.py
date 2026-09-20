"""Record a 3D model candidate for a class, with its licence, so we can credit it.

Sketchfab URLs are looked up through the public API, so the author and licence are
recorded as the site states them rather than from memory.

    python training/add_model.py mine_roller https://sketchfab.com/3d-models/...-b845931a1b...
    python training/add_model.py ta-ta <url> --note "closest AT-AT so far" --status downloaded

Writes models/registry.json (machine-readable) and models/<class>/credits.txt (the
attribution we owe). Meshes stay out of git; these two files do not.
"""

import argparse
import json
import re
import sys
import urllib.request
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))  # dtos.py lives in src/

from dtos import OBJECT_CLASSES  # noqa: E402

SKETCHFAB_UID = re.compile(r'sketchfab\.com/3d-models/[^/]*-([0-9a-f]{32})')


def sketchfab_details(url: str):
    match = SKETCHFAB_UID.search(url)
    if not match:
        return {}
    api = f'https://api.sketchfab.com/v3/models/{match.group(1)}'
    try:
        with urllib.request.urlopen(api, timeout=30) as response:
            data = json.load(response)
    except Exception as exc:
        print(f'could not reach the Sketchfab API ({exc}); recording the URL only')
        return {}
    licence = data.get('license') or {}
    return {
        'title': data.get('name'),
        'author': (data.get('user') or {}).get('displayName'),
        'author_profile': (data.get('user') or {}).get('profileUrl'),
        'licence': licence.get('label'),
        'licence_requirements': licence.get('requirements'),
        'downloadable': data.get('isDownloadable'),
        'faces': data.get('faceCount'),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('class_name', choices=sorted(OBJECT_CLASSES))
    parser.add_argument('url')
    parser.add_argument('--note', default='')
    parser.add_argument('--status', default='candidate', choices=['candidate', 'downloaded', 'reference', 'rejected'])
    # Only Sketchfab can be looked up automatically; for CGTrader, Fab, Poly Pizza and the
    # rest, copy what the page says. The licence is the part worth getting right.
    parser.add_argument('--title', default='')
    parser.add_argument('--author', default='')
    parser.add_argument('--licence', default='')
    args = parser.parse_args()

    entry = {'url': args.url, 'status': args.status, 'added': date.today().isoformat()}
    if args.note:
        entry['note'] = args.note
    entry.update(sketchfab_details(args.url))
    for field, value in (('title', args.title), ('author', args.author), ('licence', args.licence)):
        if value:
            entry[field] = value
    if not entry.get('licence'):
        print('WARNING: no licence recorded. Pass --licence with what the page states.')

    registry_path = ROOT / 'models' / 'registry.json'
    registry = json.loads(registry_path.read_text()) if registry_path.exists() else {}
    entries = [e for e in registry.get(args.class_name, []) if e['url'] != args.url]
    entries.append(entry)
    registry[args.class_name] = entries
    registry_path.parent.mkdir(parents=True, exist_ok=True)
    registry_path.write_text(json.dumps(registry, indent=1, sort_keys=True))

    class_dir = ROOT / 'models' / args.class_name
    class_dir.mkdir(parents=True, exist_ok=True)
    lines = [f'{args.class_name} - 3D model sources', '']
    for e in registry[args.class_name]:
        lines += [
            f"{e.get('title') or '(untitled)'} [{e['status']}]",
            f"  {e['url']}",
            f"  author:  {e.get('author') or '?'}",
            f"  licence: {e.get('licence') or '?'} - {e.get('licence_requirements') or 'check the page'}",
        ]
        if e.get('note'):
            lines.append(f"  note:    {e['note']}")
        lines.append('')
    (class_dir / 'credits.txt').write_text('\n'.join(lines))

    print(f"{args.class_name}: {entry.get('title') or args.url}")
    print(f"  licence: {entry.get('licence') or 'unknown'} ({entry['status']})")
    print(f"  -> {registry_path.relative_to(ROOT)} and models/{args.class_name}/credits.txt")


if __name__ == '__main__':
    main()
