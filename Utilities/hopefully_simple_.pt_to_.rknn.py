from ultralytics import YOLO

model = YOLO("YoloModels/v26/nano/test/color-3.1-v26.pt")

model.export(format='rknn', name="rk3588", data="data.yaml")