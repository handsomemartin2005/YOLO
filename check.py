# -*- coding: utf-8 -*-
"""
功能：
1. 查看你改完模型后的 Params(M)
2. 查看 GFLOPs(G)
3. 测试纯推理 FPS
4. 自动汇总为 CSV

适用环境：
conda activate yoloGPU
python D:/code/yolov8/check_model_stats.py
"""

from multiprocessing import freeze_support
import os
import csv
import copy
import time
import traceback
from pathlib import Path

import torch
from ultralytics import YOLO

# =========================
# 可选依赖：thop
# 用于估算 GFLOPs
# 如果没装，也能运行，只是 GFLOPs 会显示为 "-"
# 安装方法：pip install thop
# =========================
try:
    from thop import profile as thop_profile
    HAS_THOP = True
except Exception:
    HAS_THOP = False


# =========================
# 你的路径配置
# =========================
MODEL_YAML = r"D:\code\yolov8\ultralytics\cfg\models\v8\yolov8n-full-innov.yaml"
DATA_YAML = r"D:\code\yolov8\datasets\data.yaml"

BEST_PT_LIST = [
    r"D:\code\yolov8\runs\detect\full_innov_640_01\weights\best.pt",
    r"D:\code\yolov8\runs\detect\full_innov_640_02\weights\best.pt",
    r"D:\code\yolov8\runs\detect\full_innov_640_03\weights\best.pt",
    r"D:\code\yolov8\runs\detect\full_innov_640_04\weights\best.pt",
    r"D:\code\yolov8\runs\detect\full_innov_640_05\weights\best.pt",
]

# 结果保存路径
SAVE_CSV = r"D:\code\yolov8\runs\detect\model_stats_summary.csv"

# =========================
# 测速配置
# =========================
IMGSZ = 640
BATCH = 1
WARMUP = 50
RUNS = 200
USE_HALF = True   # 只有 CUDA 下有效，CPU 会自动忽略

# True：输出 yaml 和所有存在的 best.pt
# False：只输出 best.pt
INCLUDE_YAML = True


def get_device():
    """自动选择设备。"""
    if torch.cuda.is_available():
        return torch.device("cuda:0")
    return torch.device("cpu")


def count_params_m(model: torch.nn.Module) -> float:
    """统计参数量，单位 M。"""
    return sum(p.numel() for p in model.parameters()) / 1e6


def calc_gflops(model: torch.nn.Module, imgsz: int, device: torch.device):
    """
    用 THOP 估算 GFLOPs。
    常见近似：GFLOPs ≈ MACs * 2 / 1e9
    """
    if not HAS_THOP:
        return None

    try:
        model_copy = copy.deepcopy(model).to(device).eval()
        x = torch.randn(1, 3, imgsz, imgsz, device=device).float()
        model_copy = model_copy.float()

        macs, _ = thop_profile(model_copy, inputs=(x,), verbose=False)
        gflops = macs * 2 / 1e9
        return gflops
    except Exception as e:
        print(f"[GFLOPs] 计算失败：{e}")
        return None


def benchmark_fps(
    model: torch.nn.Module,
    imgsz: int,
    device: torch.device,
    batch: int = 1,
    warmup: int = 50,
    runs: int = 200,
    use_half: bool = False
):
    """
    测纯前向 FPS。
    注意：
    1. 这是“纯模型前向”速度
    2. 不包含真实图片读取、resize、NMS、保存结果等流程
    3. 更适合比较你改前改后的结构开销
    """
    model = model.to(device).eval()

    if use_half and device.type == "cuda":
        model = model.half()
        x = torch.randn(batch, 3, imgsz, imgsz, device=device).half()
    else:
        model = model.float()
        x = torch.randn(batch, 3, imgsz, imgsz, device=device).float()

    with torch.no_grad():
        # 预热
        for _ in range(warmup):
            _ = model(x)

        if device.type == "cuda":
            torch.cuda.synchronize()

        start = time.perf_counter()
        for _ in range(runs):
            _ = model(x)
        if device.type == "cuda":
            torch.cuda.synchronize()
        end = time.perf_counter()

    total_time = end - start
    avg_time_per_batch = total_time / runs
    avg_time_per_image = avg_time_per_batch / batch

    fps = 1.0 / avg_time_per_image
    ms_per_image = avg_time_per_image * 1000.0
    return fps, ms_per_image


def collect_model_info(model_path: str, device: torch.device):
    """
    对单个模型路径收集：
    Params(M), GFLOPs(G), FPS, ms/images
    """
    print("\n" + "=" * 90)
    print(f"正在分析模型：{model_path}")
    print("=" * 90)

    result = {
        "model_name": Path(model_path).name,
        "model_path": model_path,
        "exists": "Yes" if os.path.exists(model_path) else "No",
        "params_m": "-",
        "gflops_g": "-",
        "fps": "-",
        "ms_per_image": "-",
        "imgsz": IMGSZ,
        "batch": BATCH,
        "device": str(device),
    }

    if not os.path.exists(model_path):
        print("文件不存在，跳过。")
        return result

    try:
        # 加载 Ultralytics 模型
        yolo = YOLO(model_path)
        net = yolo.model

        # 官方 summary
        print("\n[Ultralytics summary]")
        try:
            yolo.info(verbose=True)
        except Exception as e:
            print(f"info() 打印失败：{e}")

        # Params
        params_m = count_params_m(net)
        result["params_m"] = f"{params_m:.4f}"
        print(f"\nParams(M): {result['params_m']}")

        # GFLOPs
        gflops = calc_gflops(net, IMGSZ, device)
        if gflops is not None:
            result["gflops_g"] = f"{gflops:.4f}"
        print(f"GFLOPs(G): {result['gflops_g']}")

        # FPS
        fps, ms = benchmark_fps(
            model=net,
            imgsz=IMGSZ,
            device=device,
            batch=BATCH,
            warmup=WARMUP,
            runs=RUNS,
            use_half=USE_HALF
        )
        result["fps"] = f"{fps:.4f}"
        result["ms_per_image"] = f"{ms:.4f}"

        print(f"FPS: {result['fps']}")
        print(f"ms/images: {result['ms_per_image']}")

        return result

    except Exception as e:
        print(f"[错误] 模型分析失败：{e}")
        traceback.print_exc()
        return result


def save_to_csv(rows, save_path):
    """保存结果到 CSV。"""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    fieldnames = [
        "model_name",
        "model_path",
        "exists",
        "params_m",
        "gflops_g",
        "fps",
        "ms_per_image",
        "imgsz",
        "batch",
        "device",
    ]

    with open(save_path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print("\n" + "#" * 90)
    print(f"结果已保存到：{save_path}")
    print("#" * 90)


def print_summary(rows):
    """在终端打印汇总表。"""
    print("\n" + "#" * 90)
    print("最终汇总")
    print("#" * 90)

    header = f"{'模型名':<25} {'Params(M)':<12} {'GFLOPs(G)':<12} {'FPS':<12} {'ms/images':<12}"
    print(header)
    print("-" * len(header))

    for r in rows:
        print(
            f"{r['model_name']:<25} "
            f"{r['params_m']:<12} "
            f"{r['gflops_g']:<12} "
            f"{r['fps']:<12} "
            f"{r['ms_per_image']:<12}"
        )


def main():
    print("torch version:", torch.__version__)
    print("cuda available:", torch.cuda.is_available())
    print("cuda count:", torch.cuda.device_count())
    print("torch cuda version:", torch.version.cuda)
    print("HAS_THOP:", HAS_THOP)

    device = get_device()
    print("using device:", device)

    model_paths = []

    if INCLUDE_YAML:
        model_paths.append(MODEL_YAML)

    for pt in BEST_PT_LIST:
        model_paths.append(pt)

    rows = []
    for model_path in model_paths:
        row = collect_model_info(model_path, device)
        rows.append(row)

        if device.type == "cuda":
            torch.cuda.empty_cache()
        time.sleep(1)

    print_summary(rows)
    save_to_csv(rows, SAVE_CSV)


if __name__ == "__main__":
    freeze_support()
    main()