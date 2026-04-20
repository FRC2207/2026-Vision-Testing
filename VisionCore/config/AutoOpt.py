import platform
import subprocess
import os

def run_cmd(cmd):
    try:
        result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        return result.stdout.lower()
    except:
        return ""

def get_system_info():
    return {
        "cpu": platform.processor().lower(),
        "arch": platform.machine().lower(),
        "os": platform.system().lower()
    }

def has_nvidia():
    return os.system("nvidia-smi >nul 2>&1") == 0

def has_amd_gpu():
    out = run_cmd(["lspci"])
    return "amd" in out or "radeon" in out

def has_intel():
    return "intel" in platform.processor().lower()

def has_arm():
    return "arm" in platform.machine().lower() or "aarch" in platform.machine().lower()

def has_edge_tpu():
    out = run_cmd(["lsusb"])
    return "google" in out and "coral" in out

def has_rockchip_npu():
    return os.path.exists("/dev/rknpu")

def has_hailo_npu():
    return os.path.exists("/dev/hailo")

def has_apple_silicon():
    return platform.system() == "darwin" and "arm" in platform.machine().lower()

def has_intel_vpu():
    out = run_cmd(["lsusb"])
    return "movidius" in out

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