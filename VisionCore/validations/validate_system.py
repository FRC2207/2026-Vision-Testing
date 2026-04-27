# This file is gonna validate evtyihng
import os
import re
from pathlib import Path
import logging
from ez import unit_tests

# First validate file setup

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

pattern = re.compile(
    r"^YoloModels/"
    r"v\d+/" # version like v8, v11, v26
    r"(nano|medium|large)/" # model size folder
    r"(color|gray)-\d+/" # filename prefix folder part
    r".+\.(pt|onnx|tflite|rknn)$" # file + extension
)

def is_valid_model_path(path: str) -> bool:
    # normalize slashes
    path = path.replace("\\", "/")
    return bool(pattern.match(path))

def validate_model_files():
    model_dir = Path("YoloModels")
    if not model_dir.exists():
        raise FileNotFoundError("YoloModels directory not found.")
    
    for root, _, files in os.walk(model_dir):
        for file in files:
            full_path = os.path.join(root, file)
            if not is_valid_model_path(full_path):
                raise ValueError(f"Invalid model file path: {full_path}")
    logger.info("All model file paths are valid.")

# Second validate config files
def validate_config_files():
    config_dir = Path("config")
    if not config_dir.exists():
        raise FileNotFoundError("config directory not found.")
    
    for root, _, files in os.walk(config_dir):
        for file in files:
            if not file.endswith(".json"):
                raise ValueError(f"Invalid config file: {file}. Only .json files are allowed.")
    logger.info("All config files are valid.")

# Third run unit tests
def run_unit_tests():
    logger.info("Running unit tests...")
    if not unit_tests():
        raise RuntimeError("Unit tests failed.")
    logger.info("All unit tests passed successfully.")

######### This gfunctio nis to get a human readable string of like things recommend to make the system run better that I cant auto do yet
def get_recommendations():
    reccommendations = []

    return reccommendations

# Main validation function
def validate_system():
    try:
        validate_model_files()
        validate_config_files()
        run_unit_tests()
        logger.info("System validation successful. All checks passed.")
        return True
    except Exception as e:
        logger.error(f"System validation failed: {e}")
        return False