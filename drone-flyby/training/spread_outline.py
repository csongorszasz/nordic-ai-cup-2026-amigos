"""Copy one hand-drawn sprite outline onto the rest of its class.

    python training/spread_outline.py helicopter            # from the class's newest hand outline
    python training/spread_outline.py ta-ta --from ta-ta/f003_5.png --force

The cut-outs of a class all come from the same object in one flight, a frame or two apart, so
one outline fits them all once it is placed on each crop's own box (the supplied annotation).
The outline is mapped from the reference's box to each target's box (a scale and a shift), then
written into the sprite's alpha exactly as the review page does; the GrabCut original is kept in
sprites/_grabcut/. Sprites already outlined by hand are left alone unless --force. Redraw any
that come out wrong on the review page (http://localhost:8765).
"""

import argparse
import json
import shutil
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from utils import load_frame  # noqa: E402

SPRITES = ROOT / 'sprites'
MANUAL = SPRITES / 'manual.json'
BACKUP = SPRITES / '_grabcut'


def align(source_entry, target_entry):
    """A small correction (shift, turn, scale) from the reference crop to the target's, or None.

    The box-to-box mapping is right to a few pixels, but a tall object leans differently across
    the frame, so the outline is matched to the target's own pixels (ECC on the crop, which is
    the box with context around it).
    """
    src = load_frame(source_entry['frame'])
    dst = load_frame(target_entry['frame'])
    sx1, sy1, sx2, sy2 = source_entry['bbox']
    tx1, ty1, tx2, ty2 = target_entry['bbox']
    pad = 16
    a = cv2.cvtColor(src[max(sy1 - pad, 0):sy2 + pad, max(sx1 - pad, 0):sx2 + pad], cv2.COLOR_BGR2GRAY)
    b = cv2.cvtColor(dst[max(ty1 - pad, 0):ty2 + pad, max(tx1 - pad, 0):tx2 + pad], cv2.COLOR_BGR2GRAY)
    if a.size < 64 or b.size < 64:
        return None
    b = cv2.resize(b, (a.shape[1], a.shape[0]))
    warp = np.eye(2, 3, dtype=np.float32)
    try:
        cv2.findTransformECC(a.astype(np.float32), b.astype(np.float32), warp, cv2.MOTION_EUCLIDEAN,
                             (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 200, 1e-5), None, 5)
    except cv2.error:
        return None
    return warp


def map_polygon(polygon, source_box, target_box, warp=None):
    """The outline of one crop placed on another, its box the same fraction of the way across."""
    sx1, sy1, sx2, sy2 = source_box
    tx1, ty1, tx2, ty2 = target_box
    kx, ky = (tx2 - tx1) / max(sx2 - sx1, 1), (ty2 - ty1) / max(sy2 - sy1, 1)
    if warp is None:
        return [[tx1 + (x - sx1) * kx, ty1 + (y - sy1) * ky] for x, y in polygon]
    pad = 16
    out = []
    for x, y in polygon:   # into the reference crop, through the match, back out in the target's frame
        u, v = x - (sx1 - pad), y - (sy1 - pad)
        u2 = warp[0, 0] * u + warp[0, 1] * v + warp[0, 2]
        v2 = warp[1, 0] * u + warp[1, 1] * v + warp[1, 2]
        out.append([float((tx1 - pad) + u2 * kx), float((ty1 - pad) + v2 * ky)])
    return out


def write_alpha(entry, edits):
    """Replay the reference's edits (replace / add / subtract) into this sprite's alpha."""
    path, backup = SPRITES / entry['file'], BACKUP / entry['file']
    if not backup.exists():
        backup.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, backup)
    x1, y1, x2, y2 = entry['bbox']
    rgb = load_frame(entry['frame'])[y1:y2, x1:x2]
    alpha = cv2.imread(str(backup), cv2.IMREAD_UNCHANGED)[:, :, 3].copy()   # start from GrabCut's
    for edit in edits:
        if edit['mode'] == 'replace':
            alpha[:] = 0
        pts = np.round(np.array(edit['polygon']) - [x1, y1]).astype(np.int32)
        cv2.fillPoly(alpha, [pts], 0 if edit['mode'] == 'subtract' else 255)
    cv2.imwrite(str(path), np.dstack([rgb, alpha]))
    return float((alpha > 0).mean())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('class_name')
    parser.add_argument('--from', dest='source', help='sprite file to copy the outline from (default: the class\'s)')
    parser.add_argument('--force', action='store_true', help='also replace outlines drawn by hand')
    parser.add_argument('--no-align', action='store_true', help='box-to-box only, without matching the pixels')
    args = parser.parse_args()

    index = {e['file']: e for e in json.loads((SPRITES / 'index.json').read_text())}
    manual = json.loads(MANUAL.read_text()) if MANUAL.exists() else {}
    drawn = {f: m for f, m in manual.items() if (m.get('polygon') or m.get('edits'))
             and index.get(f, {}).get('class') == args.class_name}
    if not drawn:
        raise SystemExit(f'no hand-drawn outline for {args.class_name}: draw one at http://localhost:8765')
    source = args.source or max(drawn, key=lambda f: len(drawn[f].get('edits') or [1]))
    edits = drawn[source].get('edits') or [{'mode': 'replace', 'polygon': drawn[source]['polygon']}]
    print(f'{source}: {len(edits)} outline edit(s) -> the rest of {args.class_name}')

    for file, entry in index.items():
        if entry['class'] != args.class_name or file == source:
            continue
        if file in drawn and not args.force:
            print(f'  {file}: already drawn by hand, left alone')
            continue
        warp = None if args.no_align else align(index[source], entry)
        mapped = [{'mode': e['mode'], 'polygon': map_polygon(e['polygon'], index[source]['bbox'], entry['bbox'], warp)}
                  for e in edits]
        share = write_alpha(entry, mapped)
        manual[file] = {'status': 'manual', 'polygon': mapped[0]['polygon'], 'edits': mapped, 'from': source}
        print(f'  {file}: {share:.0%} of the crop kept' + ('' if warp is not None else ', boxes only (no pixel match)'))
    MANUAL.write_text(json.dumps(manual, indent=1))


if __name__ == '__main__':
    main()
