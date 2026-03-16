# pt_to_wtv.py — one script, does everything
from ultralytics import YOLO
import os
from rknn.api import RKNN

pt_dir = os.path.dirname("YoloModels/v26/nano/testing/color-3.1-v26.pt")
model = YOLO("YoloModels/v26/nano/testing/color-3.1-v26.pt")
model.export(format="rknn", name="rk3588", imgsz=(640, 640))

pt_basename = os.path.splitext(os.path.basename("YoloModels/v26/nano/testing/color-3.1-v26.pt"))[0]
onnx_path = os.path.join(pt_dir, f"{pt_basename}.onnx")

rknn = RKNN()
rknn.config(target_platform='rk3588', mean_values=[[0,0,0]], std_values=[[255,255,255]])
rknn.load_onnx(model=onnx_path)
rknn.build(do_quantization=True, dataset="Images/RknnDataset/dataset.txt")
rknn.export_rknn("YoloModels/v26/nano/testing/color-3.1-v26.rknn")
rknn.release()

print(f"Done: {'YoloModels/v26/nano/testing/color-3.1-v26.pt'}")