from ultralytics import YOLO

model = YOLO(rf"YoloModels\v8_or_v11\gray-3.0.pt", task="detect")

model.export(format='onnx', opset=12, imgsz=640, simplify=True, dynamic=True)
# model.export(format="openvino", half=True, dynamic=False, imgsz=320) # if you want to export the model to ONNX format, use opset=12

print("Model converted to wtv i did")