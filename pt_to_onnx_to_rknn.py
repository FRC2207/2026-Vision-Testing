# This whole file is commented out but in "theory" it works

# import os
# import torch
# import numpy as np
# from rknn.api import RKNN
# from ultralytics import YOLO

# # Enter all ur values
# PT_MODEL = "YoloModels/v8_or_v11/test/gray-3.0.pt"
# DATASET_PATH = "dataset.txt"

# IMG_SIZE = 640 # Has to match what was used training the yolo model
# MEAN_VALUES = [[0, 0, 0]] # i have no clue what this is
# STD_VALUES = [[255, 255, 255]] # i have no cleu what this is

# print("[1/2] Exporting .pt → .onnx")

# model = YOLO(PT_MODEL)
# model.export(
#     format="onnx",
#     imgsz=IMG_SIZE,
#     opset=12, # most widely compatible opset
#     simplify=True,
#     dynamic=False
# )

# if not os.path.exists(PT_MODEL.replace(".pt", ".onnx")):
#     raise RuntimeError("onnx export failed, file not found")

# print("\n[2/2] Converting onnx to rknn")

# rknn = RKNN(verbose=True)

# rknn.config(
#     mean_values=MEAN_VALUES,
#     std_values=STD_VALUES,
#     optimization_level=3,
#     quantized_dtype = "w8a8", # idk but docs said best one for fps
#     target_platform="rk3588" # works for our pi
# )

# ret = rknn.load_onnx(model=PT_MODEL.replace(".pt", ".onnx"))
# if ret != 0:
#     raise RuntimeError("Failed to load onnx model")

# ret = rknn.build(
#     do_quantization=True,
#     dataset=DATASET_PATH
# )
# if ret != 0:
#     raise RuntimeError("rknn build failed")

# print("\n[3/] Exporting .rknn")

# ret = rknn.export_rknn(PT_MODEL.replace(".pt", ".rknn"))
# if ret != 0:
#     raise RuntimeError("Failed to export rknn")

# rknn.release()
# print("rknn model created (hopefully)")
