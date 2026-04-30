from ultralytics import YOLO
import torch

'''
python D:\code\yolov8\train_ppap_hg.py
'''

print("torch version:", torch.__version__)
print("cuda available:", torch.cuda.is_available())
print("cuda count:", torch.cuda.device_count())

model = YOLO(r"D:\code\yolov8\ultralytics\cfg\models\v8\yolov8n-ppapema-hg.yaml")
model.load(r"D:\code\yolov8\yolov8n.pt")

model.train(
    data=r"D:\code\yolov8\datasets\data.yaml",
    imgsz=640,
    epochs=1,
    batch=2,
    device="cpu",
    workers=0
)