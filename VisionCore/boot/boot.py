from config.AutoOpt import recommend_format
from validations.validate_system import validate_system
from config.VisionCoreConfig import VisionCoreConfig
import os
import logging
from pathlib import Path
import subprocess

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

FORMAT_EXTENSIONS = {
    "onnx":      ".onnx",
    "openvino":  ".xml",
    "rknn":      ".rknn",
    "tflite":    ".tflite",
    "coreml":    ".mlpackage",
}

def search_for_config() -> str:
    config_dir = Path("config")
    if not config_dir.exists():
        raise FileNotFoundError("config directory not found.")

    config_files = [
        os.path.join(root, file)
        for root, _, files in os.walk(config_dir)
        for file in files
        if file.endswith(".json")
    ]

    if not config_files:
        raise FileNotFoundError("No config files found in the config directory.")

    logger.info(f"Found config files: {config_files}. Using: {config_files[0]}")
    return config_files[0]

def on_boot():
    logger.info("Starting VisionCore boot sequence...")

    # 1. Validate system
    if not validate_system():
        raise RuntimeError("System validation failed. Aborting boot.")
    logger.info("System validation passed.")

    # 2. Load config
    config_file = search_for_config()
    config = VisionCoreConfig(config_file)
    logger.info(f"Loaded config from {config_file}.")

    # 3. Auto-optimization
    if config.get("auto_opt", False):
        best_format = recommend_format()
        logger.info(f"Auto-opt enabled. Recommended format: {best_format}")

        extension = FORMAT_EXTENSIONS.get(best_format)
        if not extension:
            raise ValueError(f"No extension mapping for format: {best_format}")

        model_dir = Path("YoloModels")
        optimized_models = list(model_dir.rglob(f"*{extension}"))

        if optimized_models:
            chosen = str(optimized_models[0])
            logger.info(f"Found optimized model(s): {[str(m) for m in optimized_models]}. Using: {chosen}")
            config.set("model_path", chosen)
        else:
            logger.warning(f"No {best_format} models found in YoloModels/. Falling back to config model path.")
    else:
        logger.info("Auto-opt disabled. Using model path from config.")

    model_path = config.get("model_path")
    if not model_path or not Path(model_path).exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")

    logger.info(f"Boot sequence complete. Model: {model_path}")
    logger.info(f"Starting main application with config: {config_file}")

    try:
        subprocess.run(["python", "install.py"], check=True)
    except subprocess.CalledProcessError as e:
        logger.error(f"Failed to run install.py: {e}")
        raise RuntimeError("Boot failed during service installation.")

on_boot()