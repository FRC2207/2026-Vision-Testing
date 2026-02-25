from ultralytics import YOLO
import os

os.chdir("YoloModels/v26/nano/test")

model = YOLO("color-3.1-v26.pt")

model.export(
    format="rknn",
    name="rk3588",
    data="../../../data.yaml"
)