import os
import re
import sys
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

_MODEL_PATTERN = re.compile(
    r"^YoloModels/"
    r"v\d+/"
    r"(nano|medium|large)/"
    r"(color|gray)-\d+\.\d+.*"
    r"\.(pt|onnx|tflite|rknn)$"
)


def is_valid_model_path(path: str) -> bool:
    return bool(_MODEL_PATTERN.match(path.replace("\\", "/")))

def validate_model_files() -> None:
    model_dir = Path("YoloModels")
    if not model_dir.exists():
        raise FileNotFoundError("YoloModels directory not found.")

    for root, _, files in os.walk(model_dir):
        for file in files:
            full_path = os.path.join(root, file)
            if not is_valid_model_path(full_path):
                raise ValueError(f"Invalid model file path: {full_path}")

    logger.info("All model file paths are valid.")

def validate_config_files() -> None:
    config_dir = Path("config")
    if not config_dir.exists():
        raise FileNotFoundError("config directory not found.")

    for root, _, files in os.walk(config_dir):
        for file in files:
            if not file.endswith(".json"):
                raise ValueError(f"Invalid config file: {file}. Only .json files are allowed.")

    logger.info("All config files are valid.")

def run_unit_tests() -> None:
    logger.info("Running unit tests…")

    # Ensure the repo root is on sys.path so absolute imports work.
    repo_root = Path(__file__).resolve().parents[2]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

    import unittest
    loader = unittest.TestLoader()
    # The test file is unit_tests.py, not test.py.
    suite = loader.discover(
        start_dir=str(Path(__file__).parent),
        pattern="unit_tests.py",
    )
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    if not result.wasSuccessful():
        raise RuntimeError("Unit tests failed.")

    logger.info("All unit tests passed.")

def get_recommendations() -> list[str]:
    # Work on later :)
    return []

def validate_system() -> bool:
    try:
        validate_model_files()
        validate_config_files()
        run_unit_tests()
        logger.info("System validation successful.")
        return True
    except Exception as e:
        logger.error("System validation failed: %s", e)
        return False