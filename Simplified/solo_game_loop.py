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
from Classes.FuelTracker import FuelTracker
from Classes.Metrics import Metrics
import signal
from rknnlite.api import RKNNLite # No error handling :)

shutdown_event = threading.Event()
signal.signal(signal.SIGINT,  lambda sig, frame: shutdown_event.set())
signal.signal(signal.SIGTERM, lambda sig, frame: shutdown_event.set())

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    filemode="w",
    filename="log.txt",
)

logger = logging.getLogger(__name__)

camera = Camera(
    constants.CONFIG.camera_config("Arducam"),
    constants.CONFIG,
    RKNNLite.NPU_CORE_0_1,
)

metrics = Metrics()

if constants["app_mode"]:
    camera_app = CameraApp()
    threading.Thread(target=camera_app.run, daemon=True).start()

if constants["use_network_tables"]:
    network_handler = NetworkTableHandler(constants["network_tables_ip"])

def numpy_to_fuel_list(fuel_positions: np.ndarray) -> list[Fuel]:
    return [Fuel(p[0], p[1]) for p in fuel_positions]

if __name__ == "__main__":
    try:
        logger.info("Starting simplified, single-camera loop.")

        try:
            raw_fuel_positions, annotated_frame = camera.run()
            fuel_list = numpy_to_fuel_list(raw_fuel_positions)
        except Exception as e:
            logger.warning(f"Warm-up run failed: {e}")
            fuel_list = []
            annotated_frame = None

        planner = PathPlanner(constants.CONFIG)
        fuel_tracker = FuelTracker(constants.CONFIG)

        while not shutdown_event.is_set():
            start_time = time.perf_counter()
            camera_lag_s = camera.get_frame_age()

            vision_start = time.perf_counter()
            try:
                raw_fuel_positions, annotated_frame = camera.run()
                fuel_list = numpy_to_fuel_list(raw_fuel_positions)
            except Exception as e:
                fuel_list = []
                annotated_frame = None
                logger.exception(f"Vision exception: {e}")
            vision_s = time.perf_counter() - vision_start

            robot_pose = network_handler.get_robot_pose()
            robot_x = robot_pose.X()
            robot_y = robot_pose.Y()
            
            fuel_list = fuel_tracker.update(fuel_list, np.array(robot_x, robot_y))

            flask_s = None
            if constants["app_mode"]:
                if annotated_frame is None:
                    logger.warning("Frame not returned from camera.run()")
                else:
                    flask_start = time.perf_counter()
                    camera_app.set_frame(annotated_frame)
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
            if constants["use_network_tables"]:
                network_start = time.perf_counter()
                network_handler.send_fuel_list(
                    fuel_list, "vision_data", "VisionData"
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
            print(f"\rFPS: {1/loop_s:.1f}      ", end="")
    finally:
        camera.destroy()
        metrics.destroy()