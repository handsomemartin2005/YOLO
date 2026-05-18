# -*- coding: utf-8 -*-
"""Train the full innovation YOLOv8n model on GC10-DET."""

from __future__ import annotations

import argparse
import time
import traceback
from multiprocessing import freeze_support
from pathlib import Path

import torch
from ultralytics import YOLO


ROOT = Path(__file__).resolve().parent
DEFAULT_MODEL = ROOT / "ultralytics" / "cfg" / "models" / "v8" / "yolov8n-full-innov.yaml"
DEFAULT_DATA = ROOT / "GC10dataset" / "data.yaml"
DEFAULT_PROJECT = ROOT / "runs" / "detect"


def parse_pretrained(value: str):
    lower = str(value).strip().lower()
    if lower in {"true", "1", "yes", "y"}:
        return True
    if lower in {"false", "0", "no", "n"}:
        return False
    if lower in {"none", "null"}:
        return None
    return value


def parse_cache(value: str):
    lower = str(value).strip().lower()
    if lower in {"false", "0", "no", "none"}:
        return False
    if lower in {"true", "1", "yes", "y"}:
        return True
    if lower in {"ram", "disk"}:
        return lower
    raise argparse.ArgumentTypeError("--cache must be one of none, false, true, ram, or disk")


def get_device(device: str):
    if device != "auto":
        return device
    return 0 if torch.cuda.is_available() else "cpu"


def run_name(base_name: str, run_idx: int, total_runs: int) -> str:
    return base_name if total_runs == 1 else f"{base_name}_{run_idx:02d}"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train full innovation YOLOv8n on GC10-DET.")
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--project", type=Path, default=DEFAULT_PROJECT)
    parser.add_argument("--name", default="gc10_full_innov")
    parser.add_argument("--runs", type=int, default=1, help="Repeat complete training runs with incremented seeds.")
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--pretrained", default="True")
    parser.add_argument("--save-period", type=int, default=10)
    parser.add_argument("--patience", type=int, default=None)
    parser.add_argument("--cache", type=parse_cache, default=False)
    parser.add_argument("--exist-ok", action="store_true")
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--deterministic", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--optimizer", default="auto")
    parser.add_argument("--cos-lr", action="store_true")
    parser.add_argument("--check-only", action="store_true", help="Validate paths and model construction without training.")
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if args.runs < 1:
        raise ValueError("--runs must be >= 1")
    if not args.model.exists():
        raise FileNotFoundError(f"Model config not found: {args.model}")
    if not args.data.exists():
        raise FileNotFoundError(f"Dataset config not found: {args.data}")


def train_once(args: argparse.Namespace, idx: int, device) -> None:
    name = run_name(args.name, idx, args.runs)
    print("\n" + "=" * 80)
    print(f"Starting GC10 training run {idx}/{args.runs}: {name}")
    print("=" * 80)

    model = YOLO(str(args.model))
    model.train(
        data=str(args.data),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        workers=args.workers,
        device=device,
        project=str(args.project),
        name=name,
        exist_ok=args.exist_ok,
        pretrained=parse_pretrained(args.pretrained),
        save=True,
        save_period=args.save_period,
        cache=args.cache,
        patience=args.patience if args.patience is not None else args.epochs,
        seed=args.seed + idx - 1,
        deterministic=args.deterministic,
        verbose=True,
        resume=False,
        amp=args.amp,
        optimizer=args.optimizer,
        cos_lr=args.cos_lr,
    )


def main() -> None:
    args = parse_args()
    validate_args(args)

    device = get_device(args.device)
    print("torch version:", torch.__version__)
    print("cuda available:", torch.cuda.is_available())
    print("cuda count:", torch.cuda.device_count())
    print("model:", args.model)
    print("data:", args.data)
    print("project:", args.project)
    print("device:", device)

    if args.check_only:
        model = YOLO(str(args.model))
        model.info()
        print("check-only passed")
        return

    args.project.mkdir(parents=True, exist_ok=True)
    success_runs: list[int] = []
    failed_runs: list[int] = []

    for idx in range(1, args.runs + 1):
        try:
            train_once(args, idx, device)
            success_runs.append(idx)
        except Exception as exc:
            print("\n" + "!" * 80)
            print(f"GC10 training run {idx} failed: {exc}")
            traceback.print_exc()
            print("!" * 80 + "\n")
            failed_runs.append(idx)
        finally:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            if idx < args.runs:
                time.sleep(5)

    print("\n" + "#" * 80)
    print("GC10 training finished")
    print("success runs:", success_runs)
    print("failed runs:", failed_runs)
    print("#" * 80)

    if failed_runs:
        raise SystemExit(1)


if __name__ == "__main__":
    freeze_support()
    main()
