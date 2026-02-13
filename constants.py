import numpy as np

DEBUG_MODE = True
AUTO_ADJUST = True

#######################################################################
# Custom PathPlanner Stuff
#######################################################################

# Robot
STARTING_POSITION = np.array([0, 0])
OBSTACLES = []

# Fuel constants
BALL_DIAMETER = 6

# Field Constants
MIN_X = -50
MAX_X = 50
MIN_Y = 0
MAX_Y = 100

# Other
ANIMATE = True
COLLECTION_PRIORITY = 0.7  # 0.0 = smoothest, 1.0 = hit all balls exactly
SMOOTHING_FACTOR = 20  # Lower = closer to waypoints, Higher = smoother
DEGREE = 2

# Spline Stuff/DBSCAN Stuff
ELIPSON = 20
MIN_SAMPLES = 3

PRESET = 0
if PRESET == 1:
    # Waypoint
    COLLECTION_PRIORITY = 1
    SMOOTHING_FACTOR = 0
    DEGREE = 1
elif PRESET == 2:
    # Smoothest
    COLLECTION_PRIORITY = 0
    SMOOTHING_FACTOR = 50
    DEGREE = 5
elif PRESET == 3:
    # Optimal??
    COLLECTION_PRIORITY = 0.5
    SMOOTHING_FACTOR = 35
    DEGREE = 3

AUTO_ADJUST_DEGREE = True

#######################################################################
# Camera stuff
#######################################################################
CAMERA_FOV = 74.9
KNOWN_CALIBRATION_DISTANCE = 12
BALL_D_INCHES = 5.90551
KNOWN_CALIBRATION_PIXEL_HEIGHT = 334 # This is guestimated from averaging data and hand tweaking it. Averaging ended at 292.5555555555556
YOLO_MODEL_FILE = "YoloModels/gray-3.0.pt"
GRAYSCALE = True