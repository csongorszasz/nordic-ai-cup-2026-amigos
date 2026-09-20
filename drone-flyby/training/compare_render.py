"""Side-by-side viewer: real frames next to the same frames with the matched render pasted in.

For every Helsinki frame that contains the class, it writes a crop of the real frame and
the same crop with the fitted render replacing the object, plus an index.html to step
through them. Hold the space bar to flip between the two in place, which makes any
difference in orientation, size or colour obvious.

    python training/render_models.py hangar --drop Cube.032 Lamps   # fit first
    python training/compare_render.py hangar                         # then compare
    # then open http://localhost:8765 (training/sprite_review/server.py): sidebar -> Model renders

A third panel paints the model with the real colours baked from the cut-outs
(render_models.bake_texture), which carries camouflage the colour fit cannot.

Objects cut by the frame edge are fitted like the rest when most of them is visible.
Otherwise they reuse the pose of the nearest fitted frame and are placed by the label's
uncut edges, which are exact.
"""

import argparse
import html
import json
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import render_models as rm  # noqa: E402
from utils import load_frame  # noqa: E402

CONTEXT = 1.1         # crop margin around the object, as a fraction of its size on each side
MOSTLY_VISIBLE = 0.8  # a cut object is still fitted when its box keeps this share of the full size
EDGE = 2              # px: a label edge this close to the frame border was cut by it


def paste(frame, rgba, x, y):
    """Alpha-blend an RGBA patch into a BGR frame at (x, y), clipped to the frame."""
    out = frame.copy()
    h, w = rgba.shape[:2]
    x0, y0 = max(x, 0), max(y, 0)
    x1, y1 = min(x + w, frame.shape[1]), min(y + h, frame.shape[0])
    if x1 <= x0 or y1 <= y0:
        return out
    patch = rgba[y0 - y:y1 - y, x0 - x:x1 - x]
    alpha = patch[:, :, 3:4].astype(np.float32) / 255.0
    bgr = cv2.cvtColor(patch[:, :, :3], cv2.COLOR_RGB2BGR).astype(np.float32)
    region = out[y0:y1, x0:x1].astype(np.float32)
    out[y0:y1, x0:x1] = (bgr * alpha + region * (1 - alpha)).astype(np.uint8)
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('class_name')
    parser.add_argument('--model', help='folder under models/<class>/ (default: best mean IoU in match.json)')
    parser.add_argument('--scale', type=int, default=2, help='display magnification in the page')
    parser.add_argument('--out', default=str(ROOT / 'datasets' / 'model_match'))
    args = parser.parse_args()

    class_dir = ROOT / 'models' / args.class_name
    match_path = class_dir / 'match.json'
    if not match_path.exists():
        raise SystemExit(f'no {match_path}; run training/render_models.py {args.class_name} first')
    matches = json.loads(match_path.read_text())
    name = args.model or max(matches, key=lambda k: matches[k]['mean_iou'])
    match = matches[name]
    mesh_path = ROOT / match['mesh']
    print(f'{args.class_name}: model "{name}" ({mesh_path.name}), dropped {match.get("dropped_parts", [])}')

    meshes, height = rm.load_normalised(mesh_path, match.get('up', 'auto'), match.get('dropped_parts', []),
                                         match.get('thicken', 1.0))
    renderer = rm.Renderer(meshes, height)
    yaws = list(range(0, 360, 5))

    index = json.loads((ROOT / 'sprites' / 'index.json').read_text())
    entries = sorted((e for e in index if e['class'] == args.class_name), key=lambda e: e['frame'])
    area = lambda e: (e['bbox'][2] - e['bbox'][0]) * (e['bbox'][3] - e['bbox'][1])
    full = [area(e) for e in entries if not e.get('truncated')]
    if not full:
        raise SystemExit('no complete (untruncated) examples to fit against')
    full_area = float(np.median(full))

    fits = {}
    for entry in entries:
        if entry.get('truncated') and area(entry) < MOSTLY_VISIBLE * full_area:
            continue
        sprite = cv2.cvtColor(cv2.imread(str(ROOT / 'sprites' / entry['file']), cv2.IMREAD_UNCHANGED),
                              cv2.COLOR_BGRA2RGBA)
        fit = rm.fit_sprite(renderer, sprite, yaws, match.get('tilts', rm.TILTS), gaps=match.get('close_gaps', 0))
        if fit:
            fit['gains'], fit['offsets'], fit['colour_error'] = rm.fit_colour(fit['placed'], sprite)
            fits[entry['frame']] = (sprite, fit)

    texture = rm.bake_texture(renderer, list(fits.values()))
    painted = rm.Renderer(rm.textured_meshes(meshes, texture), height, flat=True) if texture is not None else None

    out = Path(args.out) / f'{args.class_name}_compare'
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    for entry in entries:
        frame_no = entry['frame']
        x1, y1, x2, y2 = entry['bbox']
        if frame_no in fits:
            sprite, fit = fits[frame_no]
            note = f'fitted here: IoU {fit["iou"]:.2f}, colour error {fit["colour_error"]:.0f}/255'
        else:
            source = min(fits, key=lambda f: abs(f - frame_no))
            sprite, fit = fits[source]
            note = f'cut by the frame edge: pose from frame {source}, placed by the label\'s uncut edges'

        def draw(r):
            return rm.place(r.render(fit['yaw'], fit['tilt'], fit['lean'])[0], fit['transform'], sprite.shape)
        patch = rm.apply_colour(draw(renderer), fit['gains'], fit['offsets'])
        paint = draw(painted) if painted else None
        if frame_no in fits:
            x, y = x1, y1
        else:
            # The label is exact where it is not cut: align the patch's silhouette there.
            ys, xs = np.nonzero(patch[:, :, 3] > 64)
            x = x1 - xs.min() if x1 > EDGE else x2 - 1 - xs.max()
            y = y1 - ys.min() if y1 > EDGE else y2 - 1 - ys.max()
        h, w = patch.shape[:2]

        frame = load_frame(frame_no)
        cx, cy = x + w / 2, y + h / 2
        half_w, half_h = w * (0.5 + CONTEXT), h * (0.5 + CONTEXT)
        X0 = int(max(cx - half_w, 0)); Y0 = int(max(cy - half_h, 0))
        X1 = int(min(cx + half_w, frame.shape[1])); Y1 = int(min(cy + half_h, frame.shape[0]))

        cv2.imwrite(str(out / f'f{frame_no:03d}_real.png'), frame[Y0:Y1, X0:X1])
        cv2.imwrite(str(out / f'f{frame_no:03d}_render.png'), paste(frame, patch, int(x), int(y))[Y0:Y1, X0:X1])
        if paint is not None:
            cv2.imwrite(str(out / f'f{frame_no:03d}_paint.png'), paste(frame, paint, int(x), int(y))[Y0:Y1, X0:X1])
        rows.append({'frame': frame_no, 'note': note, 'yaw': fit['yaw'], 'tilt': fit['tilt'], 'lean': fit['lean']})
        print(f'  frame {frame_no:3d}: {note}')
    # The renderers are not closed: with two alive, pyrender fails deleting either one's
    # GL programs. The process ends right after anyway.

    page = PAGE.replace('__TITLE__', html.escape(f'{args.class_name}: real vs {name}')) \
               .replace('__ROWS__', json.dumps(rows)) \
               .replace('__SCALE__', str(args.scale))
    (out / 'index.html').write_text(page)
    print(f'open {out / "index.html"}')


PAGE = """<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<title>__TITLE__</title>
<style>
  :root { --bg: #16181c; --panel: #1f2228; --text: #e6e6e6; --muted: #8a8f99; --accent: #58a6ff; }
  a { color: var(--accent); text-decoration: none; }
  body { margin: 0; background: var(--bg); color: var(--text); font: 14px/1.4 system-ui, sans-serif; }
  header { padding: 12px 16px; display: flex; gap: 16px; align-items: baseline; flex-wrap: wrap; }
  h1 { font-size: 16px; margin: 0; }
  .muted { color: var(--muted); }
  kbd { background: var(--panel); border: 1px solid #3a3f48; border-radius: 4px; padding: 1px 6px; }
  main { display: flex; gap: 16px; padding: 0 16px 16px; flex-wrap: wrap; }
  figure { margin: 0; background: var(--panel); padding: 8px; border-radius: 8px; }
  figcaption { color: var(--muted); margin-bottom: 6px; }
  img { image-rendering: pixelated; display: block; }
  .flip figcaption { color: var(--accent); }
  nav { padding: 0 16px 16px; display: flex; gap: 6px; flex-wrap: wrap; }
  nav button { background: var(--panel); color: var(--text); border: 1px solid #3a3f48; border-radius: 6px; padding: 4px 10px; cursor: pointer; }
  nav button.on { border-color: var(--accent); color: var(--accent); }
</style></head><body>
<header>
  <a href="/">&larr; sprite review</a>
  <h1>__TITLE__</h1>
  <span id="info" class="muted"></span>
  <span class="muted"><kbd>&larr;</kbd> <kbd>&rarr;</kbd> frame &nbsp; hold <kbd>space</kbd> to show the real frame in the render panels</span>
</header>
<nav id="nav"></nav>
<main>
  <figure><figcaption>real frame</figcaption><img id="real"></figure>
  <figure class="flip"><figcaption>render, colour fit</figcaption><img id="render"></figure>
  <figure class="flip"><figcaption>render, real colours baked on</figcaption><img id="paint"></figure>
</main>
<script>
const rows = __ROWS__, scale = __SCALE__;
let i = 0;
const real = document.getElementById('real'), render = document.getElementById('render'), paint = document.getElementById('paint');
const pad = n => String(n).padStart(3, '0');
function show() {
  const r = rows[i], f = 'f' + pad(r.frame);
  real.src = f + '_real.png'; render.src = f + '_render.png'; paint.src = f + '_paint.png';
  document.getElementById('info').textContent =
    `frame ${r.frame} (${i + 1}/${rows.length}) - yaw ${r.yaw}, tilt ${r.tilt} toward ${r.lean} deg - ${r.note}`;
  document.querySelectorAll('nav button').forEach((b, k) => b.classList.toggle('on', k === i));
}
for (const img of [real, render, paint]) img.onload = () => { img.width = img.naturalWidth * scale; };
rows.forEach((r, k) => {
  const b = document.createElement('button'); b.textContent = r.frame;
  b.onclick = () => { i = k; show(); }; document.getElementById('nav').appendChild(b);
});
const captions = [...document.querySelectorAll('.flip figcaption')];
const labels = captions.map(c => c.textContent);
function flip(toReal) {
  const f = 'f' + pad(rows[i].frame);
  render.src = toReal ? f + '_real.png' : f + '_render.png';
  paint.src = toReal ? f + '_real.png' : f + '_paint.png';
  captions.forEach((c, k) => c.textContent = toReal ? 'real frame (release space)' : labels[k]);
}
addEventListener('keydown', e => {
  if (e.key === 'ArrowRight') { i = Math.min(i + 1, rows.length - 1); show(); }
  if (e.key === 'ArrowLeft') { i = Math.max(i - 1, 0); show(); }
  if (e.key === ' ') { e.preventDefault(); if (!e.repeat) flip(true); }
});
addEventListener('keyup', e => { if (e.key === ' ') flip(false); });
show();
</script></body></html>
"""


if __name__ == '__main__':
    main()
