# -*- coding: utf-8 -*-
from multiprocessing import freeze_support
import os
import time
import traceback
import torch
from ultralytics import YOLO

"""
用法：
conda activate yoloGPU
python D:/code/yolov8/train_full_innov.py
"""


def run_one_exp(exp_idx: int,
                model_yaml: str,
                data_yaml: str,
                project_dir: str,
                base_name: str,
                epochs: int,
                imgsz: int,
                batch: int,
                workers: int,
                device,
                seed: int):
    """
    跑一轮完整训练
    """
    run_name = f"{base_name}_{exp_idx:02d}"
    print("\n" + "=" * 80)
    print(f"开始第 {exp_idx} 轮完整训练: {run_name}")
    print("=" * 80)

    # 每一轮都重新实例化模型，确保是“完整重跑”
    model = YOLO(model_yaml)

    results = model.train(
        data=data_yaml,
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        workers=workers,
        device=device,
        project=project_dir,
        name=run_name,
        exist_ok=False,      # 每轮必须新建文件夹，避免覆盖
        pretrained=True,
        save=True,
        save_period=10,
        cache=False,
        patience=epochs,     # 基本不早停，让它完整跑完
        seed=seed,           # 每轮给不同 seed，更像独立重复实验
        verbose=True,
        resume=False         # 明确不是接着上一次，而是重新开始
    )

    print(f"第 {exp_idx} 轮训练完成：{run_name}")
    return results


def main():
    print("torch version:", torch.__version__)
    print("cuda available:", torch.cuda.is_available())
    print("cuda count:", torch.cuda.device_count())
    print("torch cuda version:", torch.version.cuda)

    device = 0 if torch.cuda.is_available() else "cpu"
    print("using device:", device)

    # =========================
    # 路径配置
    # =========================
    model_yaml = r"D:\code\yolov8\ultralytics\cfg\models\v8\yolov8n-full-innov.yaml"
    data_yaml = r"D:\code\yolov8\datasets\data.yaml"

    # 保存到你原来的 runs/detect 下面
    project_dir = r"D:\code\yolov8\runs\detect"
    os.makedirs(project_dir, exist_ok=True)

    # =========================
    # 训练配置
    # =========================
    total_runs = 5          # 一晚上连续完整跑 5 次，你可以改成 10
    epochs = 500            # 每一次完整训练的 epoch 数
    imgsz = 640
    batch = 32
    workers = 8

    # 实验名前缀
    base_name = "full_innov_640"

    # 基础随机种子，不同轮次自动 +1
    base_seed = 3407

    success_runs = []
    failed_runs = []

    for i in range(1, total_runs + 1):
        try:
            run_one_exp(
                exp_idx=i,
                model_yaml=model_yaml,
                data_yaml=data_yaml,
                project_dir=project_dir,
                base_name=base_name,
                epochs=epochs,
                imgsz=imgsz,
                batch=batch,
                workers=workers,
                device=device,
                seed=base_seed + i
            )
            success_runs.append(i)

            # 两轮之间稍微停一下，避免显存/文件句柄没完全释放
            torch.cuda.empty_cache()
            time.sleep(5)

        except Exception as e:
            print("\n" + "!" * 80)
            print(f"第 {i} 轮训练失败")
            print(f"错误信息: {e}")
            traceback.print_exc()
            print("!" * 80 + "\n")
            failed_runs.append(i)

            # 失败后继续跑下一轮，不中断整夜任务
            torch.cuda.empty_cache()
            time.sleep(5)

    print("\n" + "#" * 80)
    print("全部任务结束")
    print("成功轮次:", success_runs)
    print("失败轮次:", failed_runs)
    print("#" * 80)


if __name__ == "__main__":
    freeze_support()
    main()