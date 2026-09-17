# Drone-Flyby — Training and Manual Deployment Guide

This guide explains how to train the current detector baseline on the supplied Helsinki scene and how to run the API for manual submission testing.

## 1. Is the system ready for training?

Yes, for a baseline run.

The current repository already has:

- a working API endpoint in [api.py](api.py)
- a configurable pipeline in [src/core/**init**.py](src/core/__init__.py)
- a detector baseline in [src/core/detector.py](src/core/detector.py)
- a tracker in [src/core/tracker.py](src/core/tracker.py)
- a camera policy in [src/core/camera_policy.py](src/core/camera_policy.py)
- a dataset recorder in [src/offline/record_dataset.py](src/offline/record_dataset.py)
- training/export scaffolds in [src/offline/train_yolo.py](src/offline/train_yolo.py) and [src/offline/export_tensorrt.py](src/offline/export_tensorrt.py)

That is enough to train a first YOLO model and deploy it behind the API.

## 2. Recommended baseline path

For the first training pass, use:

- the supplied [data/helsinki](data/helsinki) frames as supervised labels
- a YOLOv11s-style model if Ultralytics is installed
- the current template-bank detector as a fallback if you want something deterministic before YOLO is ready

## 3. Build the training dataset

The training helper now splits the supplied Helsinki frames into train/val subsets and can also accept pseudo-labeled recordings later.

Example:

```bash
python src/offline/train_yolo.py --build-only --output-dir training_artifacts
```

This writes a YOLO dataset layout under:

- `training_artifacts/drone_flyby_dataset/images/train`
- `training_artifacts/drone_flyby_dataset/images/val`
- `training_artifacts/drone_flyby_dataset/labels/train`
- `training_artifacts/drone_flyby_dataset/labels/val`

## 4. Train YOLO

If `ultralytics` is installed, training can be run directly:

```bash
python src/offline/train_yolo.py \
  --output-dir training_artifacts \
  --helsinki-dir data/helsinki \
  --weights yolo11s.pt \
  --epochs 120 \
  --imgsz 960 \
  --batch 8 \
  --device 0
```

If you do not have `ultralytics` yet, install it in the conda environment first.

The script defaults are tuned for this small dataset and the 8 GB laptop GPU: `multi_scale`
is off, `optimizer=auto` (Ultralytics picks AdamW with an auto-tuned LR), `patience=0`
(early stopping is unsafe here because the 5-image val set improves in fits and starts), and
`epochs=120`. Useful flags: `--multi-scale`, `--optimizer`, `--lr0`, `--patience`, `--project`,
`--name`.

Optimizer choice matters a lot by model size:

- yolo11s with `optimizer=auto` reached **mAP50 0.629 / mAP50-95 0.381**.
- The same yolo11s with `optimizer=SGD lr0=0.01` collapsed (classification head never
  converged; early-stopped at epoch 21 with mAP50 0.06).
- yolo11n preferred `SGD lr0=0.01` (**mAP50 0.424**) over `auto` (mAP50 0.382).

### 4.1 Train the yolo11n detector

A smaller YOLOv11n detector is trained the same way by passing `--weights yolo11n.pt`:

```bash
python src/offline/train_yolo.py \
  --output-dir training_artifacts \
  --helsinki-dir data/helsinki \
  --weights yolo11n.pt \
  --epochs 120 \
  --imgsz 960 \
  --batch 8 \
  --device 0 \
  --name train-yolo11n
```

This writes `runs/detect/train-yolo11n/weights/best.pt`. Promote it to the weights folder:

```bash
cp runs/detect/train-yolo11n/weights/best.pt weights/yolo11n_drone_flyby.pt
```

The yolo11s baseline is untouched and can still be trained by passing `--weights yolo11s.pt`.

### 4.2 Train the yolo11s detector

The reference yolo11s run is `runs/detect/s-auto` (promoted to
`weights/yolo11s_drone_flyby.pt`); it reached **mAP50 0.629 / mAP50-95 0.381** (precision
0.63, recall 0.58). To reproduce it with the current defaults (`optimizer=auto`,
`patience=0`, `epochs=120`):

```bash
python src/offline/train_yolo.py \
  --output-dir training_artifacts \
  --helsinki-dir data/helsinki \
  --weights yolo11s.pt \
  --imgsz 960 \
  --batch 8 \
  --device 0 \
  --name yolo11s-auto
```

Promote the result to the weights folder:

```bash
cp runs/detect/yolo11s-auto/weights/best.pt weights/yolo11s_drone_flyby.pt
```

## 5. Optional validation recording

To capture frames during manual validation runs:

```bash
export DRONE_FLYBY_RECORD_VALIDATION_DATA=1
export DRONE_FLYBY_RECORD_DIR=recorded_validation_data
python src/api.py
```

The recorder will save incoming images and metadata per sequence.

## 6. Pseudo-label recorded validation data

Once recordings exist, generate pseudo-labels with the current detector baseline:

```bash
python src/offline/pseudo_label.py \
  --images-dir recorded_validation_data/<sequence_id>/images \
  --labels-dir recorded_validation_data/<sequence_id>/labels
```

Later, you can use a stronger detector backend for pseudo-label generation.

## 7. Deploy the API with a trained YOLO baseline

The API reads configuration from environment variables.

Set the detector type and weights path:

```bash
export DRONE_FLYBY_DETECTOR_TYPE=yolo_standard
export DRONE_FLYBY_YOLO_WEIGHTS_PATH=/absolute/path/to/best.pt
export DRONE_FLYBY_POLICY_TYPE=survey_zoom
export DRONE_FLYBY_TRACKER_TYPE=world_map
```

To use the trained yolo11n detector, point the weights path at the promoted file:

```bash
export DRONE_FLYBY_DETECTOR_TYPE=yolo_standard
export DRONE_FLYBY_YOLO_WEIGHTS_PATH="$(pwd)/weights/yolo11n_drone_flyby.pt"
```

Then run:

```bash
python src/api.py
```

If Ultralytics is present and the weights file exists, the API will use the YOLO backend. If Ultralytics is missing, it falls back safely to the template-bank baseline.

## 8. Manual validation loop

Point the evaluator at the running API:

```bash
python src/local_evaluator.py --url http://localhost:9053/predict --realtime
```

This is the closest local test to a manual submission flow.

## 9. Export for faster deployment

After training, you can export a TensorRT engine:

```bash
python src/offline/export_tensorrt.py --weights /absolute/path/to/best.pt --half
```

Then deploy with:

```bash
export DRONE_FLYBY_DETECTOR_TYPE=tensorrt
export DRONE_FLYBY_TRT_ENGINE_PATH=/absolute/path/to/best.engine
python src/api.py
```

## 10. Practical recommendation

For the first submission-ready baseline:

1. Train a YOLO model from the Helsinki dataset.
2. Test it manually through `local_evaluator.py`.
3. If it is worse than the deterministic baseline, switch back to `template_bank`.
4. If it is better, keep it and only tune the detector weights.

The tracker and camera policy are already in a usable state for this workflow.
