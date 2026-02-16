# this file is commented out because it will only work for x86_64 linux server stuff and it took me like 2 hours to setup but i finnally got a .rknn file. I also ran out of AI credits tryin so much random stuff


import os
from rknn.api import RKNN
import onnx
from onnxsim import simplify

ONNX_MODEL = "color-3.1-v26.onnx" # path to your YOLOv26 ONNX
RKNN_MODEL = "color-3.1-v26.rknn" # output RKNN file
INPUT_SIZE = [1, 3, 416, 416] # change if your model has different input size
CHANNEL_MEAN_VALUE = [0, 0, 0]
CHANNEL_STD_VALUE = [255, 255, 255]
QUANTIZE = True # True if you want INT8 quantization
DATASET_DIR = "Images/RknnDataset" # directory of images for quantization, defintelety needed

def detect_input_shape(onnx_model_path):
    model = onnx.load(onnx_model_path)
    input_tensor = model.graph.input[0]
    shape = [dim.dim_value for dim in input_tensor.type.tensor_type.shape.dim]

    if len(shape) != 4:
        raise RuntimeError(f"Unsupported input shape {shape}, expected 4D")

    # Detect channels first/last
    if shape[1] in [1, 3, 4]:
        return shape, "NCHW"
    else:
        # Convert NHWC to NCHW for RKNN
        return [shape[0], shape[3], shape[1], shape[2]], "NHWC"

print("[1/4] Simplifying ONNX model...")
onnx_model = onnx.load(ONNX_MODEL)
model_simp, check = simplify(onnx_model)
if not check:
    raise RuntimeError("ONNX simplification failed!")
onnx_simplified_path = "simplified.onnx"
onnx.save(model_simp, onnx_simplified_path)
print(f"Saved simplified ONNX model: {onnx_simplified_path}")

input_shape, data_format = detect_input_shape(onnx_simplified_path)
n_channels = input_shape[1]
print(f"[INFO] Input shape: {input_shape}, Channels: {n_channels}, Format: {data_format}")

# Step 2: Initialize RKNN
print("[2/4] Initializing RKNN Toolkit...")
rknn = RKNN(verbose=True)
target_platform = "rk3588" if is_arm() else "rk3588"

rknn.config(
    mean_values=[[0]*n_channels],
    std_values=[[255]*n_channels],
    target_platform=target_platform,
    optimization_level=3,
    channel_swap="RGB" if n_channels == 3 else None
)

print("[3/4] Loading ONNX model into RKNN...")
ret = rknn.load_onnx(model=onnx_simplified_path)
if ret != 0:
    raise RuntimeError(f"Failed to load ONNX model, error code {ret}")

print("[4/4] Building RKNN model...")
if QUANTIZE:
    if not DATASET_DIR:
        raise ValueError("For quantization, set DATASET_DIR to your dataset path")
    ret = rknn.build(do_quantization=True, dataset=DATASET_DIR)
else:
    ret = rknn.build(do_quantization=False)

if ret != 0:
    raise RuntimeError(f"RKNN build failed with code {ret}")

rknn.export_rknn(RKNN_MODEL)
print(f"RKNN model exported: {RKNN_MODEL}")

rknn.release()
print("Conversion complete!")
