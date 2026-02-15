import numpy as np

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
MIN_X = -50
MAX_X = 50
MIN_Y = 0
MAX_Y = 100

# DBSCAN Stuff
ELIPSON = 20
MIN_SAMPLES = 3

#######################################################################
# Camera stuff
#######################################################################
CAMERA_FOV = 74.9
KNOWN_CALIBRATION_DISTANCE = 12
BALL_D_INCHES = 5.90551
KNOWN_CALIBRATION_PIXEL_HEIGHT = 334 # This is guestimated from averaging data and hand tweaking it. Averaging ended at 292.5555555555556
YOLO_MODEL_FILE = "YoloModels/v26/nano/color-3.1-v26.rknn"
GRAYSCALE = False
NETWORKTABLES_IP = "10.22.7.2" # Pretty sure this is right