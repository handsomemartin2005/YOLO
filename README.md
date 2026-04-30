## Ideal Conferences

| Conference | Full Name | Location | Conference Dates | Deadline Time Zone |
|---|---|---|---|---|
| ACCV 2026 | Asian Conference on Computer Vision | Osaka, Japan | December 14–18, 2026 | 23:59 GMT |
| BMVC 2026 | British Machine Vision Conference | Lancaster, UK | November 23–26, 2026 | 23:59 AoE |

## BMVC 2026 Timeline

BMVC 2026 has an earlier submission deadline than ACCV 2026. The abstract deadline is in May, followed by the full paper and supplementary material deadline one week later.

| Event | Date |
|---|---|
| Abstract Submission Deadline | May 22, 2026 |
| Paper & Supplementary Material Submission Deadline | May 29, 2026 |
| Review Period Begins | June 8, 2026 |
| Reviews Due | June 22, 2026 |
| Rebuttal Period | July 3–10, 2026 |
| Author Notification | August 7, 2026 |
| Camera-ready Deadline | August 28, 2026 |
| Conference | November 23–26, 2026 |

## ACCV 2026 Timeline

ACCV 2026 has its main paper submission deadline in early July. The paper registration deadline is mandatory and comes before the full paper submission deadline.

| Event | Date |
|---|---|
| Workshop / Tutorial Proposal Deadline | June 28, 2026 |
| Paper Registration Deadline | July 3, 2026 |
| Paper Submission Deadline | July 5, 2026 |
| Supplementary Material Deadline | July 8, 2026 |
| Demo Abstract Submission Deadline | August 6, 2026 |
| Reviews Released to Authors | August 26, 2026 |
| Rebuttal Deadline | September 2, 2026 |
| Paper Decision Notification | September 20, 2026 |
| Camera-ready Deadline | October 4, 2026 |
| Workshops / Tutorials | December 14–15, 2026 |
| Main Conference | December 16–18, 2026 |

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
