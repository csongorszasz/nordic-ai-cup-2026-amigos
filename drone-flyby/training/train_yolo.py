"""Fine-tune a pretrained YOLO on the crops from make_dataset.py.

    python training/train_yolo.py                          # yolo11s, 40 epochs
    python training/train_yolo.py --model yolo11n.pt --epochs 80 --batch 16

Best weights land in runs/<name>/weights/best.pt, next to provenance.json: the git commit
(passed in by idun/submit.sh as GIT_COMMIT, since IDUN gets the code without .git), the
command and the settings, so any run can be traced back and retrained. A name that is
already taken is never overwritten: Ultralytics appends a number instead.

last.pt is saved after every epoch, so a run cut off by the job's time limit continues with

    python training/train_yolo.py --resume runs/<name>/weights/last.pt   # or: idun/submit.sh resume <name>
"""

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path

from ultralytics import YOLO

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', default=str(ROOT / 'datasets' / 'helsinki_yolo' / 'data.yaml'))
    parser.add_argument('--model', default='yolo11s.pt')
    parser.add_argument('--epochs', type=int, default=40)
    parser.add_argument('--imgsz', type=int, default=960)
    parser.add_argument('--batch', type=int, default=8)
    parser.add_argument('--name', default='yolo11s_baseline')
    parser.add_argument('--resume', metavar='LAST_PT', help="continue a cut-off run from its last.pt (with that run's settings)")
    args = parser.parse_args()

    if args.resume:
        model = YOLO(args.resume)
        model.add_callback('on_pretrain_routine_end', lambda trainer: write_provenance(trainer.save_dir, args, None))
        model.train(resume=True)
        write_provenance(model.trainer.save_dir, args, datetime.now().isoformat(timespec='seconds'))
        return

    model = YOLO(args.model)
    # Written when training starts too, so a run cut off by the time limit still says where it came from.
    model.add_callback('on_pretrain_routine_end', lambda trainer: write_provenance(trainer.save_dir, args, None))
    model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        project=str(ROOT / 'runs'),
        name=args.name,
        exist_ok=False,  # a rerun gets a new folder instead of overwriting weights we may want back
        workers=6,
        patience=15,
        # Aerial nadir view: up/down is as meaningful as left/right.
        fliplr=0.5,
        flipud=0.5,
        degrees=10.0,
        scale=0.5,
        mosaic=1.0,
        close_mosaic=5,
        plots=False,
    )
    write_provenance(model.trainer.save_dir, args, datetime.now().isoformat(timespec='seconds'))


def write_provenance(save_dir, args, finished):
    """runs/<name>/provenance.json; a resumed run keeps the original's under "first"."""
    path = Path(save_dir) / 'provenance.json'
    record = {
        'git_commit': os.environ.get('GIT_COMMIT') or git_commit(),
        'command': ' '.join(sys.argv),
        'job_command': os.environ.get('RUN_CMD'),
        'slurm_job': os.environ.get('SLURM_JOB_ID'),
        'finished': finished,
        'args': vars(args),
    }
    if args.resume and path.exists():
        old = json.loads(path.read_text())
        record['first'] = old.get('first', old) if old.get('args', {}).get('resume') else old
    path.write_text(json.dumps(record, indent=1))
    print(f'provenance -> {path}')


def git_commit() -> str:
    """Commit of the local checkout (with -dirty if it has uncommitted changes), or 'unknown'."""
    try:
        out = subprocess.run(['git', 'describe', '--always', '--dirty'], capture_output=True, text=True,
                             cwd=ROOT, check=True)
        return out.stdout.strip()
    except Exception:
        return 'unknown'


if __name__ == '__main__':
    main()
