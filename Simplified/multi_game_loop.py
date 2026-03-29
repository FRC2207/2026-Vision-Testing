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
# from rknnlite.api import RKNNLite
import cv2
import sys

shutdown_event = threading.Event()
signal.signal(signal.SIGINT,  lambda sig, frame: shutdown_event.set())
signal.signal(signal.SIGTERM, lambda sig, frame: shutdown_event.set())

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    filemode="w",
    filename="log.txt"
)

logger = logging.getLogger(__name__)

logger.info("Creating camera objects...")
camera0 = Camera(
    constants.CONFIG.camera_config("Arducam"),
    constants.CONFIG,
    # core_mask=RKNNLite.NPU_CORE_0_1
)

camera1 = Camera(
    constants.CONFIG.camera_config("Microsoft Cinema"),
    constants.CONFIG,
    # core_mask=RKNNLite.NPU_CORE_2
)

logger.info("Success!")

logger.info("Creating metrics, camera app, camera handler, and network table handlers...")
metrics = Metrics()

if constants.APP_MODE:
    camera_app = CameraApp()
    threading.Thread(target=camera_app.run, daemon=True).start()

camera_handler = MultipleCameraHandler([camera0, camera1], constants.YOLO_MODEL_FILE)

if constants.USE_NETWORK_TABLES:
    network_handler = NetworkTableHandler(constants.NETWORKTABLES_IP)

logger.info("Success!")

def numpy_to_fuel_list(fuel_positions: np.ndarray) -> list[Fuel]:
    return [Fuel(p[0], p[1]) for p in fuel_positions]

if __name__ == "__main__":
    try:
        logger.info("Starting simplified, multi-camera loop...")
        logger.info("Warming up...")

        try:
            raw_fuel_positions = camera_handler.predict()
            fuel_list = numpy_to_fuel_list(raw_fuel_positions)
        except Exception as e:
            logger.warning(f"Warm-up run failed: {e}")
            fuel_list = []

        fuel_tracker = FuelTracker(fuel_list, constants.DISTANCE_THRESHOLD)
        logger.info("Warmed up.")

        while not shutdown_event.is_set():
            start_time = time.perf_counter()
            ages = [cam.get_frame_age() for cam in camera_handler.cameras]
            camera_lag_s = sum(ages) / len(ages) if ages else 0.0
            
            vision_start = time.perf_counter()
            try:
                raw_fuel_positions = camera_handler.predict()
                fuel_list = numpy_to_fuel_list(raw_fuel_positions)
            except Exception as e:
                fuel_list = []
                logger.exception(f"Vision exception: {e}")
            vision_s = time.perf_counter() - vision_start

            fuel_list = fuel_tracker.update(fuel_list)

            flask_s = None
            if constants.APP_MODE:
                frame = camera_handler.get_combined_frame()
                if frame is None:
                    logger.warning("Combined frame unavailable.")
                else:
                    flask_start = time.perf_counter()
                    camera_app.set_frame(frame)
                    flask_s = time.perf_counter() - flask_start

            if len(fuel_list) == 0:
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
                    fuel_list, "vision_data", "VisionData"
                )

                for camera in camera_handler.cameras:
                    # Sends hopper data
                    data = camera.get_data_for_subsytem("hopper")
                    if data is not None:
                        network_handler.send_boolean(
                            data, "has_fuel", "VisionData"
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
            # print(f"\rFPS: {1/loop_s:.3f}      ", end="")
    finally:
        camera_handler.destroy()
        metrics.destroy()