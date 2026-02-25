# Game loop file that should be run on the pi for one camera
from Classes.Camera import Camera
from Classes.PathPlanner import PathPlanner
from Classes.NetworkTableHandler import NetworkTableHandler
import time
import constants
from Classes.CameraApp import CameraApp
import threading
import logging
import numpy as np
from Classes.Fuel import Fuel

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    filemode="w", # Append dont overwrite
    filename="log.txt"
)

logger = logging.getLogger(__name__)

# Camera class
camera = Camera(
    0,
    constants.CAMERA_FOV, constants.KNOWN_CALIBRATION_DISTANCE,
        constants.BALL_D_INCHES, constants.KNOWN_CALIBRATION_PIXEL_HEIGHT,
    constants.YOLO_MODEL_FILE,
    constants.CAMERA_DOWNWARD_PITCH_ANGLE, constants.CAMERA_BOT_RELATIVE_YAW,
    constants.CAMERA_HEIGHT,
    constants.CAMERA_X_OFFSET,
    constants.CAMERA_Y_OFFSET,
    grayscale=constants.GRAYSCALE,
    debug_mode=constants.DEBUG_MODE,
    subsystem="field"
)

if constants.APP_MODE:
    camera_app = CameraApp()
    threading.Thread(target=camera_app.run, daemon=True).start()

network_handler = NetworkTableHandler(constants.NETWORKTABLES_IP)

def numpy_to_fuel_list(fuel_positions):
    return [Fuel(p[0], p[1]) for p in fuel_positions]

def fuel_list_to_numpy(fuel_list: list[Fuel]):
    return np.array([fuel.get_position() for fuel in fuel_list])

if __name__ == "__main__":
    try:
        logger.info("Starting simplified loop.")

        # Create planner
        try:
            fuel_positions = fuel_list_to_numpy(camera.run())
        except:
            fuel_positions = []

        planner = PathPlanner(
            fuel_positions, constants.STARTING_POSITION,
            constants.ELIPSON, constants.MIN_SAMPLES,
            debug_mode=constants.DEBUG_MODE,
        )

        while True:
            # start_time = time.perf_counter()
            fuel_positions = camera.run()
            for fuel_position in fuel_positions:
                print(fuel_position)
            # vision_end_time = time.perf_counter()

            if len(fuel_positions) == 0:
                logger.warning("No fuel positions detected. Skipping loop iteration.")
                continue
            else:
                # logger.info(f"Detected fuels: {len(fuel_positions)}")
                pass

            if constants.APP_MODE:
                _, frame = camera.get_yolo_data()
                camera_app.set_frame(frame)

            _, fuel_positions = planner.update_fuel_positions(fuel_positions)
            network_handler.send_data(fuel_positions, "field_positions", "yolo_data")

            # end_time = time.perf_counter()
            # logger.info(f"Vision time: {vision_end_time - start_time}. Loop time: {end_time - start_time}")
    finally:
        camera.destroy()