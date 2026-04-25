import platform
import subprocess
import os

def _run(cmd):
    try:
        return subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True).stdout.lower()
    except:
        return ""

def has_nvidia():
    return os.system("nvidia-smi >nul 2>&1") == 0

def has_amd_gpu():
    return "amd" in _run(["lspci"]) or "radeon" in _run(["lspci"])

def has_intel():
    return "intel" in platform.processor().lower()

def has_arm():
    return "arm" in platform.machine().lower() or "aarch" in platform.machine().lower()

def has_edge_tpu():
    lsusb = _run(["lsusb"])
    return "18d1:9302" in lsusb or "1ac1:089a" in lsusb

def has_rockchip_npu():
    try:
        import rknnlite
        return True
    except ImportError:
        pass
    return os.path.exists("/dev/rknpu") or os.path.exists("/dev/rknpu0")

def has_hailo_npu():
    return os.path.exists("/dev/hailo0") or os.path.exists("/dev/hailo")

def has_apple_silicon():
    return platform.system().lower() == "darwin" and "arm" in platform.machine().lower()

def has_intel_vpu():
    return "movidius" in _run(["lsusb"]) or "03e7:2485" in _run(["lsusb"])

def recommend_format():
    if has_edge_tpu():
        return "tflite"
    if has_rockchip_npu():
        return "rknn"
    if has_hailo_npu():
        return "hef"
    if has_intel_vpu():
        return "openvino"
    if has_apple_silicon():
        return "coreml"
    if has_nvidia():
        return "onnx"
    if has_amd_gpu():
        return "onnx"
    if has_intel():
        return "openvino"
    if has_arm():
        return "tflite"
    return "onnx"