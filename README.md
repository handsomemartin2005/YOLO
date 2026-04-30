# YOLOv8 Industrial Defect Ablation

This repository contains a modified YOLOv8n detector for NEU-DET style industrial surface defect detection.

The current full model is `ABC`:

- `A`: PPAP-EMA, implemented in `ultralytics/nn/modules/ppap_ema_hypergraph.py`
- `B`: KDE-driven hypergraph feature fusion, implemented in `ultralytics/nn/modules/ppap_ema_hypergraph.py`
- `C`: RGCU + CLAG guided top-down fusion, implemented in `ultralytics/nn/modules/ppap_ema_hypergraph.py`

The default dataset path is:

```text
datasets/data.yaml
```

The dataset is expected to keep this layout:

```text
datasets/
  data.yaml
  images/train
  images/val
  labels/train
  labels/val
```

## Environment

Use Python 3.9-3.11. On AutoDL or another GPU server, install a CUDA-compatible PyTorch build first, then install the remaining dependencies.

Example:

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
```

Run commands from the repository root so Python imports the local modified `ultralytics` package.

## Full Model Training

The original full-model training entrypoint is:

```bash
python train_full_innov.py
```

It uses `ultralytics/cfg/models/v8/yolov8n-full-innov.yaml` and writes results under `runs/detect`.

## Ablation Training

The ablation controller is:

```bash
python ablation.py
```

By default it runs all 8 combinations:

```text
base, A, B, C, AB, AC, BC, ABC
```

Results are written under:

```text
runs/ablation
```

A summary CSV is appended at:

```text
runs/ablation/ablation_summary.csv
```

Recommended full run on one GPU:

```bash
python ablation.py --epochs 500 --imgsz 640 --batch 32 --workers 8 --device 0
```

Quick smoke test:

```bash
python ablation.py --variants base --epochs 1 --imgsz 640 --batch 2 --workers 0 --device cpu
```

Generate model YAMLs without training:

```bash
python ablation.py --dry-run
```

The generated YAML files are placed in:

```text
ultralytics/cfg/models/v8/ablation_generated
```

Use only the six middle ablations if the baseline and full model are already available:

```bash
python ablation.py --variants six --epochs 500 --imgsz 640 --batch 32 --workers 8 --device 0
```

## Notes

- Do not move `datasets/` unless you also pass `--data /path/to/data.yaml`.
- `runs/` is ignored by git because it contains training outputs and checkpoints.
- `GC10dataset/` is ignored in this minimal package because this ablation target uses `datasets/`.
