"""Fly a camera policy over a scene and save what it did, for flyby.html to play back.

The same replay as src/local_evaluator.py (offline mode: every frame, no clock), with the
same camera rules, but solution.predict is called in-process and every step is kept:
the view the policy got, what the detector saw in it, what it reported for the whole
frame (what is scored) and the view it asked for next. Settings go in as the same
environment variables the server reads (DRONE_CAMERA, DRONE_MODEL, DRONE_CONF, H_*...).

    python training/run_policy.py                                     # hybrid, Helsinki
    python training/run_policy.py --scene validation_4k --model runs/synth_400/weights/best.pt
    python training/run_policy.py --camera sweep --set H_L0_WEIGHT=4 --name sweep_test

Writes datasets/policy_traces/<name>.json. Runs are scored as the evaluator does: Helsinki
against its labels, the recorded validation flight against the hand-checked ones
(training/copenhagen_labels.py; incomplete, so real objects nobody accepted count as false).
"""

import argparse
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))

RECORDED = ROOT / 'recordings' / 'validation_4k'
OUT = ROOT / 'datasets' / 'policy_traces'


def scene_frames(scene: str) -> dict:
    """{frame number: loader} for the supplied scene or the recorded validation flight."""
    if scene == 'validation_4k':
        return {int(p.stem.split('_')[-1]): (lambda p=p: cv2.imread(str(p))) for p in sorted(RECORDED.glob('frame_*.jpg'))}
    from utils import frame_numbers, load_frame
    return {n: (lambda n=n: load_frame(n, scene)) for n in frame_numbers(scene)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--scene', default='helsinki', help="'helsinki' or 'validation_4k'")
    parser.add_argument('--camera', default='hybrid', help='DRONE_CAMERA: hybrid, sweep, l0')
    parser.add_argument('--model', help='DRONE_MODEL: detector weights (default: solution.py\'s)')
    parser.add_argument('--conf', type=float, help='DRONE_CONF')
    parser.add_argument('--set', nargs='*', default=[], metavar='KEY=VALUE', help='more settings for solution.py')
    parser.add_argument('--name', help='trace name (default: scene, camera, weights and time)')
    parser.add_argument('--lag', type=int, default=0,
                        help='frames a camera command waits before it applies (the live service: about 1)')
    parser.add_argument('--live-timing', metavar='NEXT,LATER',
                        help='camera commands as the live service applies them, replacing --lag: this share on the '
                             'next frame, this share a frame later, the rest never. Live 2026-09-19: 0.65,0.19 '
                             '(the 0.517 run) and 0.5,0.5 (the 0.357 run)')
    parser.add_argument('--timing-from', metavar='RESPONSES.jsonl',
                        help="a live run's response log (DRONE_LOG_RESPONSES): each frame's command lands as that "
                             "frame's did live (next frame, a frame later, never), replacing --lag")
    args = parser.parse_args()

    # solution.py reads its settings when imported, so they go in first.
    settings = {'DRONE_CAMERA': args.camera, 'DRONE_RECORD_DIR': ''}  # no view recording
    if args.model:
        settings['DRONE_MODEL'] = str(Path(args.model).resolve())
    if args.conf is not None:
        settings['DRONE_CONF'] = str(args.conf)
    for item in args.set:
        key, _, value = item.partition('=')
        settings[key] = value
    os.environ.update(settings)

    import solution
    from dtos import DroneFlybyPredictRequestDto
    from local_evaluator import Camera, CameraRejection, build_request, render_view, score

    seen = []  # what the detector returned for the current view
    detect = solution.run_detector

    def recording_detector(image, region):
        found = detect(image, region)
        seen[:] = [{'class': c, 'conf': round(float(s), 3), 'bbox': [round(float(v)) for v in box]} for c, s, box in found]
        return found
    solution.run_detector = recording_detector

    frames = scene_frames(args.scene)
    camera, feedback, steps, predictions, queue = Camera(), None, [], {}, []
    import random
    timing_rng = random.Random(0)
    timing_script = None
    if args.timing_from:
        live = [json.loads(line) for line in open(args.timing_from)]
        live = {l['frame']: l for l in live if l['sequence_id'] == live[-1]['sequence_id']}
        centre = lambda r: ((r[0] + r[2]) // 2, (r[1] + r[3]) // 2)
        timing_script = {}
        for f, l in live.items():
            rv = l['requested_view']
            if not rv or f + 1 not in live or f + 2 not in live:
                continue
            want = (rv['center_x'], rv['center_y'])
            timing_script[f] = ('next' if centre(live[f + 1]['region']) == want else
                                'later' if centre(live[f + 2]['region']) == want else 'never')
    started = time.monotonic()
    for index, (frame, load) in enumerate(frames.items()):
        seen.clear()
        view = (camera.resolution_level, camera.center_x, camera.center_y, list(camera.source_region))
        payload = build_request(frame, index, camera, render_view(load(), camera), feedback)
        took = time.monotonic()
        response = solution.predict(DroneFlybyPredictRequestDto.model_validate(payload))
        took = (time.monotonic() - took) * 1000
        reported = [{'class': a.object_id, 'conf': round(float(a.confidence), 3),
                     'bbox': [round(v * s) for v, s in zip(a.bbox, (3840, 2160, 3840, 2160))]} for a in response.annotations]
        predictions[frame] = [{'object_id': r['class'], 'bbox': r['bbox'], 'confidence': r['conf']} for r in reported]
        step = {'frame': frame, 'level': view[0], 'center': view[1:3], 'region': view[3], 'ms': round(took),
                'seen': list(seen), 'reported': reported, 'next': None, 'refused': None}
        state = solution._states.get(payload['sequence_id'])
        if state is not None:   # the ground motion the memory is using, px/frame
            step['motion'] = [round(float(v), 1) for v in state.velocity]
            step['motion_field'] = [[round(float(v), 2) for v in row] for row in state.field]
        feedback = None
        if response.requested_view is not None:
            step['next'] = [response.requested_view.resolution_level, response.requested_view.center_x,
                            response.requested_view.center_y]
        if timing_script is not None:   # as that frame's command landed live
            outcome = timing_script.get(frame, 'next')
            if response.requested_view is not None and outcome != 'never':
                queue.append((index + (1 if outcome == 'next' else 2), response.requested_view))
            due = [c for d, c in queue if d == index + 1]
            queue = [(d, c) for d, c in queue if d > index + 1]
            r = due[-1] if due else None
        elif args.live_timing:   # each command due on the next frame, the one after, or never; the newest due wins
            u = timing_rng.random()
            share_next, share_later = (float(v) for v in args.live_timing.split(','))
            if response.requested_view is not None and u < share_next + share_later:
                queue.append((index + (1 if u < share_next else 2), response.requested_view))
            due = [c for d, c in queue if d == index + 1]
            queue = [(d, c) for d, c in queue if d > index + 1]
            r = due[-1] if due else None
        else:
            queue.append(response.requested_view)
            r = queue.pop(0) if len(queue) > args.lag else None   # the command that applies now
        if r is not None:
            try:
                camera.apply(r.resolution_level, r.center_x, r.center_y)
            except CameraRejection as exc:
                step['refused'] = str(exc)
                feedback = {'frame': frame, 'reason': str(exc), 'requested_view': {
                    'resolution_level': r.resolution_level, 'center_x': r.center_x, 'center_y': r.center_y}}
        steps.append(step)
        print(f'frame {frame:4d} L{view[0]} ({view[1]:4d},{view[2]:4d}) seen {len(seen):2d} reported {len(reported):2d}'
              f' {took:5.0f} ms' + (f'  REFUSED: {step["refused"]}' if step['refused'] else ''))

    result = {}
    truth = None
    if args.scene == 'validation_4k':   # hand-checked labels, if built (training/copenhagen_labels.py)
        path = ROOT / 'datasets' / 'copenhagen_test' / 'labels.json'
        if path.exists():
            truth = {int(k): v for k, v in json.loads(path.read_text())['labels'].items()}
    if args.scene != 'validation_4k' or truth:
        mean, per_class = score(args.scene, predictions, truth)
        result = {'map50': round(mean, 4), 'ap50': {k: round(v, 4) for k, v in per_class.items()}}
        print(f'COCO mAP@0.50: {mean:.3f}' + ('  (Copenhagen, hand-checked labels)' if truth else ''))
        print('  ' + ', '.join(f'{k} {v:.2f}' for k, v in sorted(per_class.items(), key=lambda kv: kv[1])))
        if truth:   # tune on the first half, confirm on the second: the flight is our only test set
            for half, keep in (('tune', lambda f: f <= 125), ('check', lambda f: f > 125)):
                part = {f: v for f, v in truth.items() if keep(f)}
                result[f'map50_{half}'] = round(score(args.scene, predictions, part)[0], 4)
            print(f"  frames 1-125 (tune) {result['map50_tune']:.3f}, 126-249 (check) {result['map50_check']:.3f}")

    model_name = Path(solution.MODEL_PATH).parent.parent.name
    name = args.name or f'{args.scene}_{args.camera}_{model_name}_{datetime.now():%m%d-%H%M}'
    OUT.mkdir(parents=True, exist_ok=True)
    levels = [s['level'] for s in steps]
    (OUT / f'{name}.json').write_text(json.dumps({
        'name': name, 'scene': args.scene, 'camera': args.camera, 'model': str(solution.MODEL_PATH),
        'settings': settings, 'created': datetime.now().isoformat(timespec='seconds'),
        'seconds': round(time.monotonic() - started, 1), 'refused': sum(bool(s['refused']) for s in steps),
        'levels': {level: levels.count(level) for level in (0, 1, 2)}, **result, 'steps': steps,
    }))
    print(f'-> {OUT / name}.json  (levels L0/L1/L2 {levels.count(0)}/{levels.count(1)}/{levels.count(2)})')


if __name__ == '__main__':
    main()
