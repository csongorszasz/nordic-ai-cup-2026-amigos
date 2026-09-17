"""Fine-tune a pretrained YOLO on the crops from make_dataset.py.

    python training/train_yolo.py                          # yolo11s, 40 epochs
    python training/train_yolo.py --model yolo11n.pt --epochs 80 --batch 16

Best weights land in runs/<name>/weights/best.pt.
"""

import argparse
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

    YOLO(args.model).train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        project=str(ROOT / 'runs'),
        name=args.name,
        exist_ok=True,
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


if __name__ == '__main__':
    main()
