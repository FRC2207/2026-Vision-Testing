# Game loop file that should be run on the pi for one camera
from Classes.Camera import Camera
from Classes.PathPlanner import PathPlanner
from Classes.NetworkTableHandler import NetworkTableHandler
import time
import math
import constants
from Classes.CameraApp import CameraApp
import threading
import logging
import numpy as np
from Classes.Fuel import Fuel
from Classes.FuelTracker import FuelTracker
from Classes.Metrics import Metrics

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    filemode="w",  # Overwrite, don't append
    filename="log.txt",
)

logger = logging.getLogger(__name__)

# Camera class
camera = Camera(
    "/dev/video1",
    # "Images/1.png",
    constants.CAMERA_FOV,
    constants.KNOWN_CALIBRATION_DISTANCE,
    constants.BALL_D_INCHES,
    constants.KNOWN_CALIBRATION_PIXEL_HEIGHT,
    constants.YOLO_MODEL_FILE,
    constants.CAMERA_DOWNWARD_PITCH_ANGLE,
    constants.CAMERA_BOT_RELATIVE_YAW,
    constants.CAMERA_HEIGHT,
    constants.CAMERA_X_OFFSET,
    constants.CAMERA_Y_OFFSET,
    grayscale=constants.GRAYSCALE,
    debug_mode=constants.DEBUG_MODE,
    subsystem="field",
    input_size=(constants.YOLO_INPUT_SIZE, constants.YOLO_INPUT_SIZE),
    quantized=False,
    unit=constants.UNIT
)

metrics = Metrics()

if constants.APP_MODE:
    camera_app = CameraApp()
    threading.Thread(target=camera_app.run, daemon=True).start()

if constants.USE_NETWORK_TABLES:
    network_handler = NetworkTableHandler(constants.NETWORKTABLES_IP)

def numpy_to_fuel_list(fuel_positions):
    return [Fuel(p[0], p[1]) for p in fuel_positions]


def fuel_list_to_numpy(fuel_list: list[Fuel]):
    return np.array([fuel.get_position() for fuel in fuel_list])


if __name__ == "__main__":
    try:
        logger.info("Starting simplified, single-camera loop.")

        # Create planner
        try:
            raw_fuel_positions, annotated_frame = camera.run()
            fuel_positions = numpy_to_fuel_list(raw_fuel_positions)
        except:
            fuel_positions = []

        planner = PathPlanner(
            fuel_list_to_numpy(fuel_positions),
            constants.STARTING_POSITION,
            constants.ELIPSON,
            constants.MIN_SAMPLES,
            debug_mode=constants.DEBUG_MODE
        )
        fuel_tracker = FuelTracker(fuel_positions, constants.DISTANCE_THRESHOLD)

        # i = 0
        # while i < 500:
        while True:
            start_time = time.perf_counter()
            
            # Vision
            vision_start = time.perf_counter()
            try:
                raw_fuel_positions, annotated_frame = camera.run()
                fuel_positions = numpy_to_fuel_list(raw_fuel_positions)
            except Exception as e:
                fuel_positions, annotated_frame = [], None
                logger.exception(f"Exception: {e}")
            vision_s = time.perf_counter() - vision_start

            fuel_tracker.set_fuel_list(fuel_positions)
            fuel_tracker.sort()
            fuel_positions = fuel_tracker.get_fuel_list()

            camera_lag_s = camera.get_frame_age()

            for fuel_position in fuel_positions:
                x, y = fuel_position[0], fuel_position[1]
                distance = math.hypot(x, y)
                print(f"Distance: {distance}, X: {x}, Y: {y}")

            flask_s = None
            if constants.APP_MODE:
                if annotated_frame is None:
                    logger.warning("Frame not returned from camera.run()")
                else:
                    flask_start = time.perf_counter()
                    camera_app.set_frame(annotated_frame)
                    flask_s = time.perf_counter() - flask_start

            if len(fuel_positions) == 0:
                logger.warning("No fuel positions detected. Skipping loop iteration.")
                loop_s = time.perf_counter() - start_time
                metrics.record(
                    loop_s=loop_s,
                    vision_s=vision_s,
                    camera_lag_s=camera_lag_s,
                    flask_s=flask_s,
                )
                metrics.tick()
                continue
            else:
                # logger.info(f"Detected fuels: {len(fuel_positions)}")
                pass

            _, fuel_positions = planner.update_fuel_positions(
                fuel_list_to_numpy(fuel_positions)
            )
            if constants.USE_NETWORK_TABLES:
                network_handler.send_data(
                    fuel_positions, "field_positions", "yolo_data"
                )

            end_time = time.perf_counter()
            est_fps = 1 / (end_time - start_time)
            # logger.info(f"Loop run time: {end_time - start_time}. Est FPS: {est_fps}")
            # logger.info(f"Detected Fuel Positions: {[fuel_position for fuel_position in fuel_positions]}")
            # i += 1

            loop_s = time.perf_counter() - start_time
            metrics.record(
                loop_s=loop_s,
                vision_s=vision_s,
                camera_lag_s=camera_lag_s,
                flask_s=flask_s,
            )
            metrics.tick()
    finally:
        camera.destroy()
