"""Find more examples of one class in unlabelled frames.

Useful when a class has only one or two examples in the training scene and you want
to see what it really looks like, e.g. before downloading a 3D model of it.

Two methods:

``--method detector`` (default)
    Slide the fine-tuned YOLO over each frame at native resolution and keep the
    detections of one class. The model is weak on the validation scene, but it
    ranks that scene's objects far better than raw pixels do.
``--method template``
    Rotated template matching from the sprite cut-outs. Cheap and model-free, but
    normalised correlation saturates on bland ground (grass and bushes score ~0.97),
    so treat its ranking with suspicion.

    python training/find_class.py mine_roller
    python training/find_class.py hangar --method template --step 3

Writes a contact sheet of the best candidates plus a JSON list of where they are.

NOTE: the recorded validation frames are our only honest test set. Use the results
to LOOK at objects, not as labels to train on.
"""

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))  # dtos.py / utils.py live in src/


def load_templates(sprite_dir: Path, class_name: str, max_templates: int):
    index = json.loads((sprite_dir / 'index.json').read_text())
    entries = [e for e in index if e['class'] == class_name and not e.get('truncated')]
    entries.sort(key=lambda e: -(e['bbox'][2] - e['bbox'][0]) * (e['bbox'][3] - e['bbox'][1]))
    templates = []
    for entry in entries[:max_templates]:
        sprite = cv2.imread(str(sprite_dir / entry['file']), cv2.IMREAD_UNCHANGED)
        if sprite is not None and sprite.shape[2] == 4:
            templates.append((entry['file'], sprite))
    return templates


def rotations(sprite: np.ndarray, step: int, scales):
    """Rotated and rescaled copies of the sprite as (bgr, mask) pairs."""
    out = []
    for scale in scales:
        for angle in range(0, 360, step):
            h, w = sprite.shape[:2]
            matrix = cv2.getRotationMatrix2D((w / 2, h / 2), angle, scale)
            cos, sin = abs(matrix[0, 0]), abs(matrix[0, 1])
            out_w, out_h = int(w * cos + h * sin) + 2, int(w * sin + h * cos) + 2
            matrix[0, 2] += out_w / 2 - w / 2
            matrix[1, 2] += out_h / 2 - h / 2
            turned = cv2.warpAffine(sprite, matrix, (out_w, out_h), flags=cv2.INTER_LINEAR,
                                    borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0, 0))
            mask = (turned[:, :, 3] > 128).astype(np.uint8) * 255
            if mask.sum() < 255 * 16:
                continue
            out.append((turned[:, :, :3].copy(), mask))
    return out


def best_in_frame(frame: np.ndarray, templates, per_frame: int):
    """Top matches in one frame as (score, x1, y1, x2, y2)."""
    hits = []
    for patch, mask in templates:
        th, tw = patch.shape[:2]
        if th >= frame.shape[0] or tw >= frame.shape[1]:
            continue
        response = cv2.matchTemplate(frame, patch, cv2.TM_CCORR_NORMED, mask=mask)
        response[~np.isfinite(response)] = 0
        for _ in range(per_frame):
            _, score, _, location = cv2.minMaxLoc(response)
            if score <= 0:
                break
            x, y = location
            hits.append((float(score), x, y, x + tw, y + th))
            # Blank out a neighbourhood so the next peak is a different object.
            x0, y0 = max(x - tw, 0), max(y - th, 0)
            response[y0:y + th, x0:x + tw] = 0
    hits.sort(reverse=True)
    return hits[:per_frame]


def detector_hits(frame: np.ndarray, model, class_index: int, conf: float, tile: tuple, overlap: int):
    """Detections of one class in a 4K frame, from tiled inference at native scale."""
    tile_w, tile_h = tile
    height, width = frame.shape[:2]
    hits = []
    xs = list(range(0, max(width - tile_w, 0) + 1, tile_w - overlap)) or [0]
    ys = list(range(0, max(height - tile_h, 0) + 1, tile_h - overlap)) or [0]
    for y in ys:
        for x in xs:
            crop = frame[y:y + tile_h, x:x + tile_w]
            result = model.predict(crop, conf=conf, verbose=False, half=True)[0]
            for box, cls, score in zip(result.boxes.xyxy.tolist(),
                                       result.boxes.cls.tolist(),
                                       result.boxes.conf.tolist()):
                if int(cls) != class_index:
                    continue
                hits.append((float(score), int(box[0]) + x, int(box[1]) + y,
                             int(box[2]) + x, int(box[3]) + y))
    hits.sort(reverse=True)
    return hits


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('class_name')
    parser.add_argument('--frames-dir', default=str(ROOT / 'recordings' / 'validation_4k'))
    parser.add_argument('--sprites', default=str(ROOT / 'sprites'))
    parser.add_argument('--out', default=str(ROOT / 'datasets' / 'class_search'))
    parser.add_argument('--step', type=int, default=3, help='use every Nth frame')
    parser.add_argument('--angle-step', type=int, default=30)
    parser.add_argument('--scales', type=float, nargs='*', default=[0.85, 1.0, 1.2])
    parser.add_argument('--templates', type=int, default=2, help='sprite cut-outs to match with')
    parser.add_argument('--per-frame', type=int, default=2)
    parser.add_argument('--top', type=int, default=48, help='candidates on the contact sheet')
    parser.add_argument('--context', type=int, default=96, help='half-size of each crop, in px')
    parser.add_argument('--downscale', type=float, default=0.5,
                        help='template search at this fraction of full size (1.0 is exact, ~4x slower)')
    parser.add_argument('--method', default='detector', choices=['detector', 'template'])
    parser.add_argument('--model', default=str(ROOT / 'runs' / 'yolo11s_baseline' / 'weights' / 'best.pt'))
    parser.add_argument('--conf', type=float, default=0.05)
    args = parser.parse_args()

    frame_paths = sorted(Path(args.frames_dir).glob('frame_*.jpg'))[::args.step]
    if not frame_paths:
        raise SystemExit(f'no frames in {args.frames_dir}')

    if args.method == 'detector':
        from dtos import OBJECT_CLASSES
        from ultralytics import YOLO

        if args.class_name not in OBJECT_CLASSES:
            raise SystemExit(f'unknown class {args.class_name!r}')
        class_index = OBJECT_CLASSES.index(args.class_name)
        model = YOLO(args.model)
        print(f'{args.model} -> class {args.class_name} (index {class_index}), '
              f'{len(frame_paths)} frames, conf >= {args.conf}', flush=True)

        found = []
        for n, path in enumerate(frame_paths):
            frame = cv2.imread(str(path))
            for score, x1, y1, x2, y2 in detector_hits(frame, model, class_index, args.conf,
                                                       (960, 540), 120)[:args.per_frame]:
                found.append({'score': round(score, 4), 'frame': path.name, 'bbox': [x1, y1, x2, y2]})
            if (n + 1) % 10 == 0:
                best = max((f['score'] for f in found), default=0.0)
                print(f'{n + 1}/{len(frame_paths)} frames, {len(found)} hits, best {best:.3f}', flush=True)
        write_results(found, args)
        return

    templates = load_templates(Path(args.sprites), args.class_name, args.templates)
    if not templates:
        raise SystemExit(f'no usable sprites for {args.class_name!r} in {args.sprites}')
    # Matching runs on downscaled frames: the objects are still tens of pixels across and
    # masked matchTemplate over a full 4K frame costs about a second per template.
    factor = args.downscale
    scales = [scale * factor for scale in args.scales]
    turned = [t for _, sprite in templates for t in rotations(sprite, args.angle_step, scales)]
    print(f'{len(templates)} cut-outs -> {len(turned)} rotated/scaled templates '
          f'(searching at {factor:g}x)', flush=True)

    print(f'searching {len(frame_paths)} frames')

    found = []
    for n, path in enumerate(frame_paths):
        frame = cv2.imread(str(path))
        if factor != 1.0:
            frame = cv2.resize(frame, None, fx=factor, fy=factor, interpolation=cv2.INTER_AREA)
        for score, x1, y1, x2, y2 in best_in_frame(frame, turned, args.per_frame):
            box = [int(round(v / factor)) for v in (x1, y1, x2, y2)]  # back to full-frame pixels
            found.append({'score': round(score, 4), 'frame': path.name, 'bbox': box})
        if (n + 1) % 5 == 0:
            print(f'{n + 1}/{len(frame_paths)} frames, best so far {max(f["score"] for f in found):.3f}',
                  flush=True)

    write_results(found, args)


def write_results(found, args):
    """Contact sheet plus JSON of the best candidates, so they can be eyeballed."""
    found.sort(key=lambda f: -f['score'])
    found = found[:args.top]
    if not found:
        print('nothing found; try a lower --conf')
        return

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / f'{args.class_name}_candidates.json').write_text(json.dumps(found, indent=1))

    cells = []
    context = args.context
    for hit in found:
        frame = cv2.imread(str(Path(args.frames_dir) / hit['frame']))
        x1, y1, x2, y2 = hit['bbox']
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
        x0, y0 = max(cx - context, 0), max(cy - context, 0)
        crop = frame[y0:y0 + 2 * context, x0:x0 + 2 * context]
        if crop.shape[0] < 2 * context or crop.shape[1] < 2 * context:
            crop = cv2.copyMakeBorder(crop, 0, 2 * context - crop.shape[0], 0, 2 * context - crop.shape[1],
                                      cv2.BORDER_CONSTANT, value=(0, 0, 0))
        crop = cv2.resize(crop, (256, 256), interpolation=cv2.INTER_NEAREST)
        cv2.rectangle(crop, (int((x1 - x0) * 256 / (2 * context)), int((y1 - y0) * 256 / (2 * context))),
                      (int((x2 - x0) * 256 / (2 * context)), int((y2 - y0) * 256 / (2 * context))), (0, 255, 0), 1)
        label = f"{hit['score']:.3f} {hit['frame'].replace('frame_', '').replace('.jpg', '')}"
        cv2.putText(crop, label, (4, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1)
        cells.append(crop)

    columns = 8
    while len(cells) % columns:
        cells.append(np.zeros((256, 256, 3), np.uint8))
    sheet = np.vstack([np.hstack(cells[i:i + columns]) for i in range(0, len(cells), columns)])
    sheet_path = out / f'{args.class_name}_sheet.jpg'
    cv2.imwrite(str(sheet_path), sheet, [cv2.IMWRITE_JPEG_QUALITY, 92])
    print(f'{len(found)} candidates -> {sheet_path}')


if __name__ == '__main__':
    main()
