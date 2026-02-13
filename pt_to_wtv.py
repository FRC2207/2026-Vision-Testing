from ultralytics import YOLO

model = YOLO("YoloModels/v26/nano/colo-3.1-v26.pt", task="detect")

model.export(format="onnx")

print("Model converted to wtv i did")