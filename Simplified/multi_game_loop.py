# Game loop file that should be run on the pi for multiple cameras
from Classes.Camera import Camera
from Classes.NetworkTableHandler import NetworkTableHandler
import time
import constants
from Classes.CameraApp import CameraApp
import threading
import logging
import numpy as np
from Classes.Fuel import Fuel
from Classes.FuelTracker import FuelTracker
from Classes.MultipleCameraHandler import MultipleCameraHandler
from Classes.Metrics import Metrics
import signal
from rknnlite.api import RKNNLite # No error handling let it error out :)
import cv2

shutdown_event = threading.Event()
signal.signal(signal.SIGINT,  lambda sig, frame: shutdown_event.set())
signal.signal(signal.SIGTERM, lambda sig, frame: shutdown_event.set())

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    filemode="w",
    filename="log.txt",
)
logging.getLogger('rknn').setLevel(logging.WARNING)

logger = logging.getLogger(__name__)

camera0 = Camera(
    "/dev/video0",
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
    quantized=True,
    unit=constants.UNIT,
    core_mask=RKNNLite.NPU_CORE_0_1,
    fps_cap=50
)

camera1 = Camera(
    "/dev/video2",
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
    quantized=True,
    unit=constants.UNIT,
    core_mask=RKNNLite.NPU_CORE_2,
    fps_cap=30
)

metrics = Metrics()

if constants.APP_MODE:
    camera_app = CameraApp()
    threading.Thread(target=camera_app.run, daemon=True).start()

camera_handler = MultipleCameraHandler([camera0, camera1], constants.YOLO_MODEL_FILE)

if constants.USE_NETWORK_TABLES:
    network_handler = NetworkTableHandler(constants.NETWORKTABLES_IP)

def numpy_to_fuel_list(fuel_positions: np.ndarray) -> list[Fuel]:
    return [Fuel(p[0], p[1]) for p in fuel_positions]

if __name__ == "__main__":
    try:
        logger.info("Starting simplified, multi-camera loop.")

        try:
            raw_fuel_positions = camera_handler.predict()
            fuel_positions_fuel_list = numpy_to_fuel_list(raw_fuel_positions)
        except Exception as e:
            logger.warning(f"Warm-up run failed: {e}")
            fuel_positions_fuel_list = []

        fuel_tracker = FuelTracker(fuel_positions_fuel_list, constants.DISTANCE_THRESHOLD)

        while not shutdown_event.is_set():
            start_time = time.perf_counter()
            camera_lag_s = (camera0.get_frame_age() + camera1.get_frame_age()) / 2
            vision_start = time.perf_counter()
            try:
                raw_fuel_positions = camera_handler.predict()
                fuel_positions_fuel_list = numpy_to_fuel_list(raw_fuel_positions)
            except Exception as e:
                fuel_positions_fuel_list = []
                logger.exception(f"Vision exception: {e}")
            vision_s = time.perf_counter() - vision_start

            fuel_tracker.set_fuel_list(fuel_positions_fuel_list)
            fuel_tracker.sort()
            fuel_positions_fuel_list = fuel_tracker.get_fuel_list()

            flask_s = None
            if constants.APP_MODE:
                frame = camera_handler.get_combined_frame()
                if frame is None:
                    logger.warning("Combined frame unavailable.")
                else:
                    flask_start = time.perf_counter()
                    camera_app.set_frame(frame)
                    flask_s = time.perf_counter() - flask_start

            # for fuel in fuel_positions_fuel_list:
            #     print(fuel)

            if len(fuel_positions_fuel_list) == 0:
                logger.warning("No fuel positions detected. Skipping loop iteration.")
                loop_s = time.perf_counter() - start_time
                metrics.record(
                    loop_s=loop_s,
                    vision_s=vision_s,
                    camera_lag_s=camera_lag_s,
                    flask_s=flask_s,
                )
                metrics.tick()
                logger.info(f"FPS: {1/loop_s:.1f}")
                print(f"\rFPS: {1/loop_s:.3f}      ", end="")
                continue

            network_s = None
            if constants.USE_NETWORK_TABLES:
                network_start = time.perf_counter()
                network_handler.send_fuel_list(
                    fuel_positions_fuel_list, "vision_data", "VisionData"
                )
                network_s = time.perf_counter() - network_start

            loop_s = time.perf_counter() - start_time
            metrics.record(
                loop_s=loop_s,
                vision_s=vision_s,
                camera_lag_s=camera_lag_s,
                flask_s=flask_s,
                network_s=network_s
            )
            metrics.tick()
            logger.info(f"FPS: {1/loop_s:.1f}")
            print(f"\rFPS: {1/loop_s:.3f}      ", end="")
    finally:
        camera_handler.destroy()
        metrics.destroy()