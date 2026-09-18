"""Fine-tune a pretrained YOLO on the crops from make_dataset.py.

    python training/train_yolo.py                          # yolo11s, 40 epochs
    python training/train_yolo.py --model yolo11n.pt --epochs 80 --batch 16

Best weights land in runs/<name>/weights/best.pt, next to provenance.json: the git commit
(passed in by idun/submit.sh as GIT_COMMIT, since IDUN gets the code without .git), the
command and the settings, so any run can be traced back and retrained. A name that is
already taken is never overwritten: Ultralytics appends a number instead.
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
    args = parser.parse_args()

    model = YOLO(args.model)
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
    save_dir = Path(model.trainer.save_dir)
    (save_dir / 'provenance.json').write_text(json.dumps({
        'git_commit': os.environ.get('GIT_COMMIT') or git_commit(),
        'command': ' '.join(sys.argv),
        'job_command': os.environ.get('RUN_CMD'),
        'slurm_job': os.environ.get('SLURM_JOB_ID'),
        'finished': datetime.now().isoformat(timespec='seconds'),
        'args': vars(args),
    }, indent=1))
    print(f'provenance -> {save_dir / "provenance.json"}')


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
