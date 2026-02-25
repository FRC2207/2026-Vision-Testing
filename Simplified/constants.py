import numpy as np
import json

DEBUG_MODE = True
APP_MODE = True

#######################################################################
# Custom PathPlanner Stuff
#######################################################################

# Robot
STARTING_POSITION = np.array([0, 0])
DISTANCE_THRESHOLD = 5

# Fuel constants
BALL_DIAMETER = 6

# Field Constants
MIN_X = 0
MAX_X = 317
MIN_Y = 0
MAX_Y = 690.88

# DBSCAN Stuff
ELIPSON = 20
MIN_SAMPLES = 1

#######################################################################
# Camera stuff
#######################################################################
CAMERA_FOV = 74.9
KNOWN_CALIBRATION_DISTANCE = 12
BALL_D_INCHES = 5.90551
KNOWN_CALIBRATION_PIXEL_HEIGHT = 334 # This is guestimated from averaging data and hand tweaking it. Averaging ended at 292.5555555555556
YOLO_MODEL_FILE = "YoloModels/v26/nano/test/color-3.1-v26_rknn_model/color-3.1-v26-rk3588.rknn"
GRAYSCALE = False
NETWORKTABLES_IP = "10.22.7.2" # Pretty sure this is right

with open("Simplified/camera_positions.json", "r") as file:
    data = json.load(file)

for camera in data["cameras"]:
    CAMERA_DOWNWARD_PITCH_ANGLE = camera["downward_pitch"]
    CAMERA_BOT_RELATIVE_YAW = camera["robot_relative_yaw"]
    CAMERA_HEIGHT = camera["height"]
    CAMERA_X_OFFSET = camera["x"]
    CAMERA_Y_OFFSET = camera["y"]