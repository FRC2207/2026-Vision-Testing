from ultralytics import YOLO

model = YOLO("YoloModels/v26/color-3.1-v26.pt", task="detect")

model.export(format="openvino")

print("Model converted to wtv i did")