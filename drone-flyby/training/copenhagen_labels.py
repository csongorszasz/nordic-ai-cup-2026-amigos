"""Per-frame labels for the recorded Copenhagen flight, from the hand review (/review).

TEST SET ONLY: never train on these. They are how we score a detector + pipeline on
imagery it has never seen (training/run_policy.py --scene validation_4k).

Each accepted track (and each note that says the object is real) becomes one object. The
detector's own boxes along the track are unreliable frame by frame (cut by a view edge,
jumping onto a neighbour for a frame), so they are all carried into one reference frame
with the flight's homographies (reconstruct_frames.py), boxes touching the frame edge are
dropped, and the object's box is the spread of the rest (25th percentile of the top-left
corner, 75th of the bottom-right). That box is then carried to every frame of the flight,
clipped, and kept where at least MIN_VISIBLE of it is inside the frame.

    python training/copenhagen_labels.py

note_fixes.json holds what each review note means, by track id: {"status": "accepted" or
"rejected", "class": the right class, "frames": only these frames' boxes are the object,
"partial": true when the box covers only part of the object, "boxes_from": take the track's
boxes from this round's file (e.g. candidates_r2.json) when an earlier round's are loose,
"why": the note in short}. Reads datasets/copenhagen_test/{candidates_r*,candidates,decisions,note_fixes,manual_boxes}.json, writes labels.json
({frame: [{object_id, bbox}]}) next to them.
"""

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))

from copenhagen_candidates import FRAMES, H, OUT, W, iou, move  # noqa: E402
from reconstruct_frames import between  # noqa: E402

MIN_VISIBLE = 0.5    # share of the box inside the frame for the object to count as in view
EDGE = 3             # px: a box this close to the frame edge is cut by it
DUPLICATE_IOU = 0.5  # two tracks of one class overlapping this much in the reference frame are one object
MANUAL_REPLACE_IOU = 0.3  # a hand-drawn box replaces an object of its class it overlaps this much
PARTIAL_INSIDE = 0.5  # a partial track this much inside a whole object of its class is that object


def inside_share(a, b):
    """How much of box a lies inside box b."""
    w = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    h = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    return w * h / max((a[2] - a[0]) * (a[3] - a[1]), 1e-9)


def object_box(candidate, steps, ref, only_frames=None):
    """The object's box in frame `ref`, from its track's boxes (or those of `only_frames`)."""
    boxes = []
    for frame, box in zip(candidate['frames'], candidate['boxes']):
        if only_frames and frame not in only_frames:
            continue
        x1, y1, x2, y2 = box
        if x1 <= EDGE or y1 <= EDGE or x2 >= W - EDGE or y2 >= H - EDGE:
            continue
        boxes.append(move(np.array(box, float), between(steps, frame, ref)))
    if not boxes:   # always at an edge: the best box is all there is
        return np.array(candidate['best_box'], float)
    b = np.array(boxes)
    return np.array([np.percentile(b[:, 0], 25), np.percentile(b[:, 1], 25),
                     np.percentile(b[:, 2], 75), np.percentile(b[:, 3], 75)])


def main():
    # Every review round's candidates (candidates_r1.json, ... kept in git; candidates.json is the
    # round being reviewed), so decisions keep pointing at their tracks after a regeneration.
    candidates = {}
    for path in sorted(OUT.glob('candidates_r*.json')) + [OUT / 'candidates.json']:
        if path.exists():
            for c in json.loads(path.read_text())['candidates']:
                candidates.setdefault(c['id'], c)
    decisions = json.loads((OUT / 'decisions.json').read_text())
    fixes_path = OUT / 'note_fixes.json'
    fixes = json.loads(fixes_path.read_text()) if fixes_path.exists() else {}
    steps = {int(k): np.array(v) for k, v in json.loads((FRAMES / 'homographies.json').read_text()).items()}
    frames = sorted(int(p.stem.split('_')[-1]) for p in FRAMES.glob('frame_*.jpg'))

    objects, skipped = [], []
    for cid, d in decisions.items():
        fix = fixes.get(cid, {})
        status = fix.get('status', d['status'])
        if status == 'note':
            skipped.append((cid, 'note not read yet: add it to note_fixes.json'))
            continue
        if status != 'accepted':
            continue
        if cid not in candidates:
            skipped.append((cid, 'no longer among the candidates'))
            continue
        c = candidates[cid]
        if fix.get('boxes_from'):  # a later round tracked the object more tightly
            c = next(x for x in json.loads((OUT / fix['boxes_from']).read_text())['candidates'] if x['id'] == cid)
        cls = fix.get('class') or d.get('class') or c['class']
        ref = c['best_frame']
        objects.append({'id': cid, 'class': cls, 'ref': ref, 'box': object_box(c, steps, ref, fix.get('frames')),
                        'partial': fix.get('partial', False)})

    # One label per object: merge tracks of the same class that are the same thing.
    # A track the review says covers only part of its object is the same object as a whole one it
    # mostly lies inside; partial tracks go last so the whole ones are there to match.
    merged = []
    for o in sorted(objects, key=lambda o: (o['partial'], -len(candidates[o['id']]['frames']))):
        def same_object(m):
            if m['class'] != o['class']:
                return False
            box = move(o['box'], between(steps, o['ref'], m['ref']))
            if o['partial']:
                return inside_share(box, m['box']) >= PARTIAL_INSIDE
            return iou(box, m['box']) >= DUPLICATE_IOU
        same = next((m for m in merged if same_object(m)), None)
        if same:
            same['merged'].append(o['id'])
        else:
            merged.append({**o, 'merged': []})

    # Boxes drawn by hand in the playback page: an object nobody's track found, or a better box
    # for one that was (it replaces any object of its class it overlaps).
    manual_path = OUT / 'manual_boxes.json'
    for mid, m in (json.loads(manual_path.read_text()) if manual_path.exists() else {}).items():
        box = np.array(m['box'], float)
        replaced = [o for o in merged if o['class'] == m['class']
                    and iou(move(o['box'], between(steps, o['ref'], m['frame'])), box) >= MANUAL_REPLACE_IOU]
        merged = [o for o in merged if o not in replaced]
        merged.append({'id': mid, 'class': m['class'], 'ref': m['frame'], 'box': box, 'partial': False,
                       'merged': [i for o in replaced for i in [o['id']] + o['merged']]})

    labels = {}
    for frame in frames:
        out = []
        for o in merged:
            x1, y1, x2, y2 = move(o['box'], between(steps, o['ref'], frame))
            area = (x2 - x1) * (y2 - y1)
            cx1, cy1, cx2, cy2 = max(x1, 0), max(y1, 0), min(x2, W), min(y2, H)
            if cx2 <= cx1 or cy2 <= cy1 or (cx2 - cx1) * (cy2 - cy1) < MIN_VISIBLE * area:
                continue
            out.append({'object_id': o['class'], 'bbox': [round(float(v), 1) for v in (cx1, cy1, cx2, cy2)],
                        'track': o['id']})
        labels[frame] = out
    (OUT / 'labels.json').write_text(json.dumps({
        'objects': [{'id': o['id'], 'class': o['class'], 'ref': o['ref'], 'box': [round(float(v)) for v in o['box']],
                     'merged': o['merged']} for o in merged],
        'labels': labels,
    }))
    per_class = {}
    for o in merged:
        per_class[o['class']] = per_class.get(o['class'], 0) + 1
    print(f'{len(merged)} objects ({len(objects) - len(merged)} duplicate tracks merged), '
          f'{sum(map(len, labels.values()))} boxes over {len(frames)} frames -> {OUT / "labels.json"}')
    print('  ' + ', '.join(f'{k} {v}' for k, v in sorted(per_class.items(), key=lambda kv: -kv[1])))
    alone = [m['id'] for m in merged if m['partial']]
    if alone:
        print(f'  partial tracks with no whole object to join, labelled with their own (partial) box: {", ".join(alone)}')
    for cid, why in skipped:
        print(f'  skipped {cid}: {why}')


if __name__ == '__main__':
    main()
