from ultralytics import YOLO

model = YOLO("YoloModels/v26/nano/testing/color-3.1-v26.pt", task="detect")

# model.export(
#     format='onnx',
#     opset=12,
#     simplify=True,
#     dynamic=False,
#     imgsz=320
# )

# model.export(format='onnx', opset=12, imgsz=640, simplify=True, dynamic=True)
# model.export(format="openvino", half=True, dynamic=False, imgsz=320) # if you want to export the model to ONNX format, use opset=12

model.export(
    format="rknn",
    imgsz=320,
)

print("Model converted to wtv i did")