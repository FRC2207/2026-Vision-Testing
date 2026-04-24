
import argparse
import logging
import os
import sys
import time
import glob
import subprocess
import platform
from pathlib import Path
from typing import Optional

#TODO: Copletely vibe coded, review

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("VisionCore.Boot")

class BootConfig:
    model_pt: Optional[str] = None # Will auto seach if none
    input_size: tuple = (320, 320)
    min_conf: float = 0.6
    quantize: bool = True

    # Camera
    camera_source: Optional[object] = None # auto detected if none
    camera_fov: float = 70.0 # degrees horizontal FOV
    camera_height: float = 0.0
    camera_pitch: float = 0.0
    camera_yaw: float = 0.0
    camera_x: float = 0.0
    camera_y: float = 0.0
    grayscale: bool = False
    fps_cap: int = 30

    # Calibration  (set these if you've run camera_calibration.py)
    calibration_pixel_size: float = 292.0  # known_calibration_pixel_height
    calibration_distance: float = 12.0    # inches
    game_piece_size: float = 5.91          # inches (ball diameter)

    # Network
    network_tables_ip: Optional[str] = None  # None = don't use NT
    use_network_tables: bool = False

    # Runtime
    run_duration_s: Optional[int] = None   # None = run forever
    debug_mode: bool = False
    app_mode: bool = False                 # Flask camera feed
    record_mode: bool = False

    # RKNN dataset for quantization (only needed if converting)
    rknn_dataset_txt: Optional[str] = None  # path to dataset.txt

    # DBSCAN noise filtering
    dbscan_epsilon: float = 0.5
    dbscan_min_samples: int = 3
    distance_threshold: float = 0.3        # metres — merge threshold in FuelTracker

def _run(cmd: list[str]) -> str:
    try:
        return subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                              text=True, timeout=5).stdout.lower()
    except Exception:
        return ""


def detect_hardware() -> dict:
    info = {
        "has_rknn_npu": os.path.exists("/dev/rknpu"),
        "has_coral_tpu": "google" in _run(["lsusb"]) and "coral" in _run(["lsusb"]),
        "has_hailo": os.path.exists("/dev/hailo"),
        "has_nvidia": _run(["nvidia-smi"]) != "" and "nvidia" in _run(["nvidia-smi"]),
        "has_intel_openvino": "intel" in platform.processor().lower(),
        "is_arm": "arm" in platform.machine().lower() or "aarch" in platform.machine().lower(),
        "platform": platform.system(),
        "cpu": platform.processor(),
        "arch": platform.machine(),
    }

    # Derive recommended format
    if info["has_rknn_npu"]:
        info["recommended_format"] = "rknn"
    elif info["has_coral_tpu"]:
        info["recommended_format"] = "tflite"
    elif info["has_hailo"]:
        info["recommended_format"] = "hef"
    elif info["has_nvidia"]:
        info["recommended_format"] = "onnx"
    elif info["has_intel_openvino"]:
        info["recommended_format"] = "openvino"
    elif info["is_arm"]:
        info["recommended_format"] = "tflite"
    else:
        info["recommended_format"] = "pt"   # CPU fallback — run ultralytics directly

    return info


def print_hardware_summary(hw: dict):
    logger.info("=" * 55)
    logger.info("  Hardware Detection")
    logger.info("=" * 55)
    logger.info(f"  Platform  : {hw['platform']} / {hw['arch']}")
    logger.info(f"  CPU       : {hw['cpu'] or 'unknown'}")
    logger.info(f"  RKNN NPU  : {'✓' if hw['has_rknn_npu'] else '✗'}")
    logger.info(f"  Coral TPU : {'✓' if hw['has_coral_tpu'] else '✗'}")
    logger.info(f"  Hailo NPU : {'✓' if hw['has_hailo'] else '✗'}")
    logger.info(f"  NVIDIA GPU: {'✓' if hw['has_nvidia'] else '✗'}")
    logger.info(f"  ➜  Recommended format: {hw['recommended_format'].upper()}")
    logger.info("=" * 55)


# ═══════════════════════════════════════════════════════════════════════════════
#  MODEL RESOLUTION  — find .pt, check cache, convert if needed
# ═══════════════════════════════════════════════════════════════════════════════

_FORMAT_EXT = {
    "rknn": ".rknn",
    "onnx": ".onnx",
    "openvino": "_openvino_model",   # ultralytics creates a folder
    "tflite": ".tflite",
    "hef": ".hef",
    "pt": ".pt",
}

# Suffix appended to the stem when we convert (so we don't overwrite the original)
_RKNN_SUFFIX = "-boot"


def _find_pt(search_dirs: list[str]) -> Optional[str]:
    """Search common locations for a .pt file."""
    patterns = [
        "*.pt",
        "YoloModels/**/*.pt",
        "models/**/*.pt",
    ]
    for d in search_dirs:
        for pat in patterns:
            matches = glob.glob(os.path.join(d, pat), recursive=True)
            if matches:
                # Prefer files with 'color' in the name, then shortest path
                matches.sort(key=lambda p: (0 if "color" in p else 1, len(p)))
                return matches[0]
    return None


def _converted_path(pt_path: str, fmt: str) -> str:
    """Return the expected output path for a converted model."""
    pt = Path(pt_path)
    ext = _FORMAT_EXT.get(fmt, f".{fmt}")
    if fmt == "openvino":
        return str(pt.parent / f"{pt.stem}{ext}")
    return str(pt.parent / f"{pt.stem}{_RKNN_SUFFIX}{ext}")


def _convert_to_rknn(pt_path: str, out_path: str, cfg: BootConfig) -> bool:
    """Export .pt → .onnx → .rknn. Returns True on success."""
    logger.info("Converting .pt → ONNX …")
    try:
        from ultralytics import YOLO
        model = YOLO(pt_path)
        model.export(
            format="onnx",
            imgsz=cfg.input_size,
            simplify=True,
            opset=12,
            dynamic=False,
        )
    except Exception as e:
        logger.error(f"ONNX export failed: {e}")
        return False

    pt = Path(pt_path)
    onnx_path = str(pt.parent / f"{pt.stem}.onnx")
    if not os.path.exists(onnx_path):
        logger.error(f"Expected ONNX at {onnx_path} — not found after export.")
        return False

    logger.info("Converting ONNX → RKNN …")
    try:
        from rknn.api import RKNN
        rknn = RKNN()
        rknn.config(
            target_platform="rk3588",
            mean_values=[[0, 0, 0]],
            std_values=[[255, 255, 255]],
        )
        if rknn.load_onnx(model=onnx_path) != 0:
            raise RuntimeError("load_onnx failed")

        dataset = cfg.rknn_dataset_txt if cfg.rknn_dataset_txt else None
        if rknn.build(do_quantization=cfg.quantize, dataset=dataset) != 0:
            raise RuntimeError("build failed")

        if rknn.export_rknn(out_path) != 0:
            raise RuntimeError("export_rknn failed")

        rknn.release()
        logger.info(f"RKNN model saved → {out_path}")
        return True
    except ImportError:
        logger.warning("rknn.api not available — cannot convert to RKNN.")
        return False
    except Exception as e:
        logger.error(f"RKNN conversion failed: {e}")
        return False


def _convert_to_onnx(pt_path: str, out_path: str, cfg: BootConfig) -> bool:
    logger.info("Converting .pt → ONNX …")
    try:
        from ultralytics import YOLO
        model = YOLO(pt_path)
        model.export(
            format="onnx",
            imgsz=cfg.input_size,
            simplify=True,
            opset=12,
            dynamic=False,
        )
        pt = Path(pt_path)
        default_onnx = str(pt.parent / f"{pt.stem}.onnx")
        if default_onnx != out_path and os.path.exists(default_onnx):
            os.rename(default_onnx, out_path)
        logger.info(f"ONNX model saved → {out_path}")
        return True
    except Exception as e:
        logger.error(f"ONNX export failed: {e}")
        return False


def _convert_to_openvino(pt_path: str, out_path: str, cfg: BootConfig) -> bool:
    logger.info("Converting .pt → OpenVINO …")
    try:
        from ultralytics import YOLO
        model = YOLO(pt_path)
        model.export(
            format="openvino",
            imgsz=cfg.input_size,
            half=True,
            dynamic=False,
        )
        logger.info(f"OpenVINO model saved → {out_path}")
        return True
    except Exception as e:
        logger.error(f"OpenVINO export failed: {e}")
        return False


def resolve_model(cfg: BootConfig, hw: dict) -> tuple[str, str]:
    """
    Returns (model_path, model_type) where model_type is one of:
    'rknn', 'onnx', 'openvino', 'pt'
    """
    fmt = hw["recommended_format"]

    # ── Find source .pt ───────────────────────────────────────────────────────
    pt_path = cfg.model_pt
    if pt_path and not os.path.exists(pt_path):
        logger.warning(f"Specified .pt not found: {pt_path} — searching …")
        pt_path = None

    if pt_path is None:
        search_dirs = [os.getcwd(), os.path.dirname(__file__)]
        pt_path = _find_pt(search_dirs)
        if pt_path:
            logger.info(f"Auto-found model: {pt_path}")

    # ── Handle formats that don't need a .pt (direct model files) ─────────────
    if pt_path is None:
        # Last-ditch: look for an already-converted model
        for fmt_try in ["rknn", "onnx", "pt"]:
            ext = _FORMAT_EXT[fmt_try]
            matches = glob.glob(f"**/*{ext}", recursive=True)
            if matches:
                logger.info(f"No .pt found — using existing {fmt_try.upper()}: {matches[0]}")
                return matches[0], fmt_try
        raise FileNotFoundError(
            "No model file found. Provide --model /path/to/model.pt or put a .pt in the working directory."
        )

    # ── For CPU/pt fallback just return the .pt directly ──────────────────────
    if fmt == "pt":
        logger.info("Running on CPU — using .pt directly (no conversion needed).")
        return pt_path, "pt"

    # ── Check if converted model already exists ────────────────────────────────
    converted = _converted_path(pt_path, fmt)

    # OpenVINO creates a directory, check for it
    if fmt == "openvino":
        converted_dir = Path(pt_path).parent / f"{Path(pt_path).stem}_openvino_model"
        if converted_dir.exists():
            logger.info(f"Reusing existing OpenVINO model: {converted_dir}")
            return str(converted_dir), "openvino"
    elif os.path.exists(converted):
        logger.info(f"Reusing existing {fmt.upper()} model: {converted}")
        return converted, fmt

    # ── Convert ───────────────────────────────────────────────────────────────
    logger.info(f"No converted model found — converting {Path(pt_path).name} → {fmt.upper()} …")
    success = False

    if fmt == "rknn":
        success = _convert_to_rknn(pt_path, converted, cfg)
        if not success:
            logger.warning("RKNN conversion failed — falling back to ONNX.")
            fmt = "onnx"
            converted = _converted_path(pt_path, "onnx")
            success = _convert_to_onnx(pt_path, converted, cfg)
        if not success:
            logger.warning("ONNX conversion also failed — falling back to .pt on CPU.")
            return pt_path, "pt"

    elif fmt == "onnx":
        success = _convert_to_onnx(pt_path, converted, cfg)
        if not success:
            logger.warning("ONNX conversion failed — falling back to .pt on CPU.")
            return pt_path, "pt"

    elif fmt == "openvino":
        success = _convert_to_openvino(pt_path, converted, cfg)
        if not success:
            logger.warning("OpenVINO conversion failed — falling back to .pt on CPU.")
            return pt_path, "pt"

    else:
        logger.warning(f"No converter implemented for {fmt} — falling back to .pt on CPU.")
        return pt_path, "pt"

    return converted, fmt


# ═══════════════════════════════════════════════════════════════════════════════
#  CAMERA DETECTION
# ═══════════════════════════════════════════════════════════════════════════════

def detect_camera(cfg: BootConfig) -> object:
    """Return a camera source (int index or str path) that is actually accessible."""
    if cfg.camera_source is not None:
        logger.info(f"Using specified camera source: {cfg.camera_source}")
        return cfg.camera_source

    import cv2

    # Try integer indices 0-3 first (USB webcams)
    for idx in range(4):
        cap = cv2.VideoCapture(idx)
        if cap.isOpened():
            cap.release()
            logger.info(f"Auto-detected camera at index {idx}")
            return idx
        cap.release()

    # Try /dev/video* on Linux
    video_devs = sorted(glob.glob("/dev/video*"))
    for dev in video_devs:
        cap = cv2.VideoCapture(dev, cv2.CAP_V4L2)
        if cap.isOpened():
            cap.release()
            logger.info(f"Auto-detected camera at {dev}")
            return dev
        cap.release()

    raise RuntimeError(
        "No camera found. Connect a camera or pass --camera <source> explicitly."
    )


# ═══════════════════════════════════════════════════════════════════════════════
#  CONFIG BUILDER  — converts BootConfig → VisionCoreConfig dict
# ═══════════════════════════════════════════════════════════════════════════════

def build_vision_config(cfg: BootConfig, model_path: str, model_type: str,
                         camera_source: object, hw: dict) -> dict:
    """Build the raw dict that VisionCoreConfig will load."""

    cam_name = "BootCamera"

    return {
        "unit": "meter",
        "dbscan": {
            "elipson": cfg.dbscan_epsilon,
            "min_samples": cfg.dbscan_min_samples,
        },
        "distance_threshold": cfg.distance_threshold,
        "network_tables_ip": cfg.network_tables_ip or "",
        "use_network_tables": cfg.use_network_tables and cfg.network_tables_ip is not None,
        "app_mode": cfg.app_mode,
        "debug_mode": cfg.debug_mode,
        "record_mode": cfg.record_mode,
        "stale_threshold": 1.0,
        "metrics": True,
        "vision_model": {
            "quantized": cfg.quantize and model_type == "rknn",
            "file_path": model_path,
            "input_size": list(cfg.input_size),
            "margin": 0,
            "min_conf": cfg.min_conf,
        },
        "camera_configs": {
            cam_name: {
                "name": cam_name,
                "x": cfg.camera_x,
                "y": cfg.camera_y,
                "height": cfg.camera_height,
                "pitch": cfg.camera_pitch,
                "yaw": cfg.camera_yaw,
                "grayscale": cfg.grayscale,
                "fps_cap": cfg.fps_cap,
                "calibration": {
                    "size": cfg.calibration_pixel_size,
                    "distance": cfg.calibration_distance,
                    "game_piece_size": cfg.game_piece_size,
                    "fov": cfg.camera_fov,
                },
                "source": camera_source,
                "subsystem": "field",
            }
        },
    }

def boot(cfg: BootConfig = None):
    if cfg is None:
        cfg = BootConfig()

    logger.info("=" * 55)
    logger.info("  VisionCore Boot")
    logger.info("=" * 55)

    # ── 1. Detect hardware ────────────────────────────────────────────────────
    logger.info("Step 1/4 — Detecting hardware …")
    hw = detect_hardware()
    print_hardware_summary(hw)

    # ── 2. Resolve model (find .pt, check cache, convert if needed) ───────────
    logger.info("Step 2/4 — Resolving model …")
    try:
        model_path, model_type = resolve_model(cfg, hw)
    except FileNotFoundError as e:
        logger.error(str(e))
        sys.exit(1)
    logger.info(f"  Model: {model_path}  ({model_type.upper()})")

    # ── 3. Detect camera ──────────────────────────────────────────────────────
    logger.info("Step 3/4 — Detecting camera …")
    try:
        camera_source = detect_camera(cfg)
    except RuntimeError as e:
        logger.error(str(e))
        sys.exit(1)
    logger.info(f"  Camera source: {camera_source}")

    # ── 4. Build config and launch VisionCore ─────────────────────────────────
    logger.info("Step 4/4 — Launching VisionCore …")
    vision_cfg_dict = build_vision_config(cfg, model_path, model_type, camera_source, hw)

    # Write a temp config file (VisionCoreConfig loads from file or dict)
    import json
    import tempfile
    tmp = tempfile.NamedTemporaryFile(
        mode="w", suffix=".json", delete=False, prefix="visioncore_boot_"
    )
    json.dump(vision_cfg_dict, tmp, indent=2)
    tmp.close()
    tmp_cfg_path = tmp.name
    logger.info(f"  Config written to: {tmp_cfg_path}")

    try:
        from VisionCore.config.VisionCoreConfig import VisionCoreConfig
        from VisionCore.vision.ObjectDetectionCamera import ObjectDetectionCamera
        from VisionCore.VisionCore import VisionCore

        # RKNN core mask (only relevant on NPU hardware)
        core_mask = None
        if model_type == "rknn":
            try:
                from rknnlite.api import RKNNLite
                core_mask = RKNNLite.NPU_CORE_0_1_2
                logger.info("  Using all 3 NPU cores (NPU_CORE_0_1_2)")
            except ImportError:
                logger.warning("  rknnlite not available — core_mask ignored.")

        vision_config = VisionCoreConfig(tmp_cfg_path)

        cam_name = "BootCamera"
        camera_kwargs = dict(
            camera_config=vision_config.camera_config(cam_name),
            config=vision_config,
        )
        if core_mask is not None:
            camera_kwargs["core_mask"] = core_mask

        camera = ObjectDetectionCamera(**camera_kwargs)

        vision = VisionCore([camera], vision_config)

        logger.info("=" * 55)
        logger.info("  VisionCore is running!")
        if cfg.run_duration_s:
            logger.info(f"  Will stop after {cfg.run_duration_s}s.")
        else:
            logger.info("  Running indefinitely — Ctrl-C to stop.")
        logger.info("=" * 55)

        vision.run(duration_s=cfg.run_duration_s)

    except ImportError as e:
        logger.error(f"VisionCore import failed: {e}")
        logger.error("Make sure you're running from the repo root and all deps are installed.")
        sys.exit(1)
    finally:
        try:
            os.unlink(tmp_cfg_path)
        except Exception:
            pass


# ═══════════════════════════════════════════════════════════════════════════════
#  CLI
# ═══════════════════════════════════════════════════════════════════════════════

def _parse_args() -> BootConfig:
    p = argparse.ArgumentParser(
        description="VisionCore automatic boot — detects hardware, converts model, and runs.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Model
    p.add_argument("--model",    default=None, help=".pt model file path. Auto-searched if omitted.")
    p.add_argument("--size",     default="320x320", help="Input size WxH, e.g. 320x320 or 640x640")
    p.add_argument("--conf",     type=float, default=0.6, help="Detection confidence threshold")
    p.add_argument("--no-quant", action="store_true", help="Disable quantization during RKNN conversion")
    p.add_argument("--dataset",  default=None, help="Path to dataset.txt for RKNN quantization")

    # Camera
    p.add_argument("--camera",   default=None, help="Camera source: int index, /dev/video0, or URL")
    p.add_argument("--fov",      type=float, default=70.0, help="Camera horizontal FOV in degrees")
    p.add_argument("--height",   type=float, default=0.2,  help="Camera height above ground (metres)")
    p.add_argument("--pitch",    type=float, default=15.0, help="Camera pitch angle down (degrees)")
    p.add_argument("--yaw",      type=float, default=0.0,  help="Camera yaw from robot forward (degrees)")
    p.add_argument("--gray",     action="store_true",       help="Treat camera as grayscale")
    p.add_argument("--fps",      type=int,   default=30,   help="Camera FPS cap")

    # Calibration
    p.add_argument("--cal-size",    type=float, default=292.0, help="Calibration pixel height")
    p.add_argument("--cal-dist",    type=float, default=12.0,  help="Calibration distance (inches)")
    p.add_argument("--piece-size",  type=float, default=5.91,  help="Game piece diameter (inches)")

    # Network Tables
    p.add_argument("--ip",       default=None, help="Network Tables server IP. Omit to disable NT.")

    # Runtime
    p.add_argument("--duration", type=int, default=None, help="Stop after N seconds. Default: run forever.")
    p.add_argument("--debug",    action="store_true", help="Enable debug mode (shows CV2 window if GUI available)")
    p.add_argument("--app",      action="store_true", help="Enable Flask camera feed app")
    p.add_argument("--record",   action="store_true", help="Record video output")

    args = p.parse_args()

    # Parse size
    try:
        w, h = map(int, args.size.lower().split("x"))
        input_size = (w, h)
    except Exception:
        logger.warning(f"Could not parse --size '{args.size}' — defaulting to 320x320")
        input_size = (320, 320)

    # Parse camera source
    camera_source = args.camera
    if camera_source is not None:
        try:
            camera_source = int(camera_source)
        except ValueError:
            pass  # Keep as string path/URL

    return BootConfig(
        model_pt=args.model,
        input_size=input_size,
        min_conf=args.conf,
        quantize=not args.no_quant,
        rknn_dataset_txt=args.dataset,

        camera_source=camera_source,
        camera_fov=args.fov,
        camera_height=args.height,
        camera_pitch=args.pitch,
        camera_yaw=args.yaw,
        grayscale=args.gray,
        fps_cap=args.fps,

        calibration_pixel_size=args.cal_size,
        calibration_distance=args.cal_dist,
        game_piece_size=args.piece_size,

        network_tables_ip=args.ip,
        use_network_tables=args.ip is not None,

        run_duration_s=args.duration,
        debug_mode=args.debug,
        app_mode=args.app,
        record_mode=args.record,
    )


if __name__ == "__main__":
    boot(_parse_args())