# Game loop file that should be run on the pi for multiple cameras
from VisionCore.vision.ObjectDetectionCamera import ObjectDetectionCamera
from VisionCore.trackers.PathPlanner import PathPlanner
from VisionCore.utilities.NetworkTableHandler import NetworkTableHandler
from VisionCore.utilities.MultipleCameraHandler import MultipleCameraHandler
from VisionCore.web.CameraApp import CameraApp
from VisionCore.trackers.FuelTracker import FuelTracker
from VisionCore.web.Metrics import Metrics
from VisionCore.web.healthReporter import HealthReporter
from VisionCore.trackers.Fuel import Fuel
import VisionCore.core.constants as constants
import time
import threading
import logging
import signal
import numpy as np
# from rknnlite.api import RKNNLite

shutdown_event = threading.Event()
signal.signal(signal.SIGINT, lambda sig, frame: shutdown_event.set())
signal.signal(signal.SIGTERM, lambda sig, frame: shutdown_event.set())

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    filemode="w",
    filename="Outputs/log.txt",
)

logger = logging.getLogger(__name__)

logger.info("Creating camera objects...")
camera0 = ObjectDetectionCamera(
    constants.CONFIG.camera_config("Arducam"),
    constants.CONFIG,
    # core_mask=RKNNLite.NPU_CORE_0_1
)

camera1 = ObjectDetectionCamera(
    constants.CONFIG.camera_config("Microsoft Cinema"),
    constants.CONFIG,
    # core_mask=RKNNLite.NPU_CORE_2
)
logger.info("Camera objects created.")

metrics = Metrics()
camera_handler = MultipleCameraHandler([camera0, camera1])

camera_app = None
health = None
if constants.CONFIG["app_mode"]:
    camera_app = CameraApp()
    threading.Thread(target=camera_app.run, daemon=True).start()
    health = HealthReporter(camera_app.app, constants.CONFIG)
    health.set_camera(camera0)  # Report primary camera for health

network_handler = None
if constants.CONFIG["use_network_tables"]:
    network_handler = NetworkTableHandler(constants.CONFIG["network_tables_ip"])
    if health:
        health.set_network_handler(network_handler)

def numpy_to_fuel_list(fuel_positions: np.ndarray) -> list[Fuel]:
    return [Fuel(p[0], p[1]) for p in fuel_positions]

def run_vision(camera_handler) -> tuple[list[Fuel], any]:
    try:
        raw_fuel_positions = camera_handler.predict()
        return numpy_to_fuel_list(raw_fuel_positions), camera_handler.get_combined_frame()
    except Exception as e:
        logger.exception(f"Vision exception: {e}")
        return [], None

if __name__ == "__main__":
    try:
        logger.info("Starting simplified, multi-camera loop.")
        logger.info("Warming up...")

        fuel_list, _ = run_vision(camera_handler)

        planner = PathPlanner(constants.CONFIG)
        fuel_tracker = FuelTracker(constants.CONFIG)

        logger.info("Warmed up.")

        while not shutdown_event.is_set():
            start_time = time.perf_counter()

            ages = [cam.get_frame_age() for cam in camera_handler.cameras]
            camera_lag_s = sum(ages) / len(ages) if ages else 0.0

            vision_start = time.perf_counter()
            fuel_list, combined_frame = run_vision(camera_handler)
            vision_s = time.perf_counter() - vision_start

            if network_handler:
                robot_pose = network_handler.get_robot_pose()
                fuel_list = fuel_tracker.update(
                    fuel_list, 
                    robot_pose.X(), 
                    robot_pose.Y(), 
                    robot_pose.rotation().radians()
                )
            else:
                fuel_list = fuel_tracker.update(fuel_list, 0, 0, 0)

            flask_s = None
            if constants.CONFIG["app_mode"]:
                if combined_frame is None:
                    logger.warning("Combined frame not returned from camera handler.")
                else:
                    flask_start = time.perf_counter()
                    camera_app.set_frame(combined_frame)
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
                print(f"\rFPS: {1/loop_s:.1f}      ", end="")
                continue

            noise_positions, fuel_list = planner.update_fuel_positions(fuel_list)

            network_s = None
            if constants.CONFIG["use_network_tables"]:
                network_start = time.perf_counter()
                network_handler.send_fuel_list(fuel_list, "vision_data", "VisionData")
                loop_s = time.perf_counter() - start_time
                network_handler.send_data(1 / loop_s, "fps", "VisionData")
                network_handler.send_data(len(fuel_list), "num_detections", "VisionData")
                network_handler.send_data(camera_lag_s, "camera_lag", "VisionData")

                for camera in camera_handler.cameras:
                    data = camera.get_data_for_subsytem("hopper")
                    if data is not None:
                        network_handler.send_boolean(data, "hopper_sees_object", "VisionData")

                network_s = time.perf_counter() - network_start

            loop_s = time.perf_counter() - start_time

            health_s = None
            if health:
                health_start = time.perf_counter()
                health.tick(
                    fps=1 / loop_s, vision_s=vision_s, detections=len(fuel_list)
                )
                health_s = time.perf_counter() - health_start

            metrics.record(
                loop_s=loop_s,
                vision_s=vision_s,
                camera_lag_s=camera_lag_s,
                flask_s=flask_s,
                network_s=network_s,
                health_s=health_s,
            )
            metrics.tick()

            logger.info(f"FPS: {1/loop_s:.1f}")
            print(f"\rFPS: {1/loop_s:.1f}      ", end="")
    finally:
        camera_handler.destroy()
        metrics.destroy()