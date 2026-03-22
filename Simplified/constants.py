import numpy as np
import json

MODE = "debug"
# "game" for game
# "test" for testing with robot or sim (change network tables IP)
# "debug" for home stuff

if MODE == "game":
    DEBUG_MODE = False
    APP_MODE = False
    USE_NETWORK_TABLES = True
    NETWORKTABLES_IP = "10.22.7.2"
elif MODE == "test":
    DEBUG_MODE = True
    APP_MODE = True
    USE_NETWORK_TABLES = True
    NETWORKTABLES_IP = "127.0.0.1"
else:
    DEBUG_MODE = True
    APP_MODE = True
    USE_NETWORK_TABLES = False
    NETWORKTABLES_IP = ""

# Fuel constants
UNIT = "inch"

# Cleaning Up Fuel DEtections Stuff
ELIPSON = 10
MIN_SAMPLES = 1
DISTANCE_THRESHOLD = 5

YOLO_INPUT_SIZE = 320
YOLO_MODEL_FILE = "YoloModels/v26/nano/model.rknn"
# YOLO_MODEL_FILE = "YoloModels/v8_or_v11/3.1-320x320/color-3.1-v11.onnx"

with open("Simplified/camera_positions.json", "r") as file:
    data = json.load(file)

CAMERA_CONFIGS = data["cameras"]