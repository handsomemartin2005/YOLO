# -*- coding: utf-8 -*-
import argparse
import os
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch

from ultralytics import YOLO
from ultralytics.nn.modules import PPAPEMA, KDEHyperGraphFusion


def preprocess_image(image_path: str, imgsz: int, device):
    img_bgr = cv2.imread(image_path)
    if img_bgr is None:
        raise FileNotFoundError(f"cannot read images: {image_path}")

    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    img_resize = cv2.resize(img_rgb, (imgsz, imgsz), interpolation=cv2.INTER_LINEAR)

    tensor = torch.from_numpy(img_resize).permute(2, 0, 1).float() / 255.0
    tensor = tensor.unsqueeze(0).to(device)
    return img_rgb, img_resize, tensor


def tensor_map_to_numpy(x: torch.Tensor) -> np.ndarray:
    """
    x: [1,1,H,W] or [B,1,H,W]
    """
    if x is None:
        raise ValueError("visualization tensor is None")
    if x.ndim != 4:
        raise ValueError(f"expected 4D tensor, got {x.shape}")
    return x[0, 0].cpu().numpy()


def collect_modules(model):
    """
    从 YOLO 模型中收集 PPAPEMA 和 KDEHyperGraphFusion
    """
    det_model = model.model
    ppap_modules = []
    hg_modules = []

    for m in det_model.model.modules():
        if isinstance(m, PPAPEMA):
            m.collect_visualization = True
            ppap_modules.append(m)
        elif isinstance(m, KDEHyperGraphFusion):
            m.collect_visualization = True
            hg_modules.append(m)

    if len(ppap_modules) == 0:
        raise RuntimeError("no PPAPEMA module found")
    if len(hg_modules) == 0:
        raise RuntimeError("no KDEHyperGraphFusion module found")

    return ppap_modules, hg_modules


def build_detection_image(model, image_path: str, imgsz: int, conf: float, device):
    results = model.predict(
        source=image_path,
        imgsz=imgsz,
        conf=conf,
        save=False,
        verbose=False,
        device=device,
    )
    det_bgr = results[0].plot()
    det_rgb = cv2.cvtColor(det_bgr, cv2.COLOR_BGR2RGB)
    return det_rgb


def save_feature_figure(
    det_rgb: np.ndarray,
    ppap_before: np.ndarray,
    ppap_after: np.ndarray,
    hg_before: np.ndarray,
    hg_after: np.ndarray,
    save_path: str,
):
    fig, axes = plt.subplots(2, 3, figsize=(14, 8))

    axes[0, 0].imshow(det_rgb)
    axes[0, 0].set_title("Image for Object Detection", fontsize=12)
    axes[0, 0].axis("off")

    axes[0, 1].imshow(ppap_before, cmap="viridis")
    axes[0, 1].set_title("Feature Maps before PPAP-EMA", fontsize=12)
    axes[0, 1].axis("off")

    axes[0, 2].imshow(ppap_after, cmap="viridis")
    axes[0, 2].set_title("Feature Maps after PPAP-EMA", fontsize=12)
    axes[0, 2].axis("off")

    axes[1, 0].imshow(det_rgb)
    axes[1, 0].set_title("Image for Object Detection", fontsize=12)
    axes[1, 0].axis("off")

    axes[1, 1].imshow(hg_before, cmap="viridis")
    axes[1, 1].set_title("Feature Maps before High-Order Learning", fontsize=12)
    axes[1, 1].axis("off")

    axes[1, 2].imshow(hg_after, cmap="viridis")
    axes[1, 2].set_title("Feature Maps after High-Order Learning", fontsize=12)
    axes[1, 2].axis("off")

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True, help="pt权重或yaml模型路径")
    parser.add_argument("--images", type=str, required=True, help="输入图像路径")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--hg-index", type=int, default=0, choices=[0, 1, 2], help="选择R3/R4/R5分支: 0/1/2")
    parser.add_argument("--save-dir", type=str, default="vis_outputs")
    args = parser.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)

    device = 0 if torch.cuda.is_available() else "cpu"
    model = YOLO(args.model)
    model.model.eval()

    ppap_modules, hg_modules = collect_modules(model)
    if args.hg_index >= len(hg_modules):
        raise ValueError(f"hg-index={args.hg_index} out of range, found {len(hg_modules)} hypergraph branches")

    # 原图与输入张量
    _, _, tensor = preprocess_image(args.images, args.imgsz, device)

    # 带检测框原图
    det_rgb = build_detection_image(model, args.images, args.imgsz, args.conf, device)

    # 手动前向，触发模块内部 latest_vis 更新
    with torch.no_grad():
        _ = model.model(tensor)

    ppap_mod = ppap_modules[0]
    hg_mod = hg_modules[args.hg_index]

    if not ppap_mod.latest_vis:
        raise RuntimeError("PPAPEMA latest_vis is empty, forward may not have been triggered")
    if not hg_mod.latest_vis:
        raise RuntimeError("KDEHyperGraphFusion latest_vis is empty, forward may not have been triggered")

    ppap_before = tensor_map_to_numpy(ppap_mod.latest_vis["input"])
    ppap_after = tensor_map_to_numpy(ppap_mod.latest_vis["output"])

    hg_before = tensor_map_to_numpy(hg_mod.latest_vis["before_hg"])
    hg_after = tensor_map_to_numpy(hg_mod.latest_vis["after_hg"])

    save_path = str(Path(args.save_dir) / f"feature_vis_hg{args.hg_index}.png")
    save_feature_figure(
        det_rgb=det_rgb,
        ppap_before=ppap_before,
        ppap_after=ppap_after,
        hg_before=hg_before,
        hg_after=hg_after,
        save_path=save_path,
    )

    print(f"[OK] visualization saved to: {save_path}")


if __name__ == "__main__":
    main()

    '''
cd /d D:\code\yolov8
python visualize_innov_features.py --model D:\code\yolov8\runs\detect\full_innov_64013\weights\best.pt --images D:\code\yolov8\datasets\images\val\crazing_1.jpg --imgsz 640 --hg-index 0 --save-dir D:\code\yolov8\runs\detect\full_innov_64013\feature_vis
python visualize_innov_features.py --model D:\code\yolov8\runs\detect\full_innov_64013\weights\best.pt --images D:\code\yolov8\datasets\images\val\crazing_1.jpg --imgsz 640 --hg-index 1 --save-dir D:\code\yolov8\runs\detect\full_innov_64013\feature_vis
python visualize_innov_features.py --model D:\code\yolov8\runs\detect\full_innov_64013\weights\best.pt --images D:\code\yolov8\datasets\images\val\crazing_1.jpg --imgsz 640 --hg-index 2 --save-dir D:\code\yolov8\runs\detect\full_innov_64013\feature_vis
    '''
