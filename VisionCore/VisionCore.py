# This is the class u call smth like VisionCore(cameras, config).run()
from VisionCore.utilities.MultipleCameraHandler import MultipleCameraHandler
from VisionCore.vision.Camera import Camera
from VisionCore.trackers.PathPlanner import PathPlanner
from VisionCore.utilities.NetworkTableHandler import NetworkTableHandler
import time
from VisionCore.web.CameraApp import CameraApp
import threading
import logging
import os
import numpy as np
from VisionCore.trackers.Fuel import Fuel
from VisionCore.trackers.FuelTracker import FuelTracker
from VisionCore.web.Metrics import Metrics
from VisionCore.web.healthReporter import HealthReporter
from VisionCore.utilities.VideoRecorder import VideoRecorder
from VisionCore.config.VisionCoreConfig import VisionCoreConfig
import signal

try:
    from rknnlite.api import RKNNLite
    RKNN_FOUND = True
except ImportError:
    RKNN_FOUND = False


class VisionCore:
    def __init__(self, cameras: list[Camera], config: VisionCoreConfig):
        self.cameras = cameras
        self.config = config
        self.shutdown_event = threading.Event()

        os.makedirs("Outputs", exist_ok=True)
        logging.basicConfig(
            level=getattr(logging, self.config.get("log_level", "INFO").upper(), logging.INFO),
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            filemode="w",
            filename=self.config.get("log_file", "Outputs/log.txt"),
        )
        self.logger = logging.getLogger(__name__)

        signal.signal(signal.SIGINT,  lambda sig, frame: self.shutdown_event.set())
        signal.signal(signal.SIGTERM, lambda sig, frame: self.shutdown_event.set())

        self.metrics = Metrics() if self.config["metrics"] else None

        self.camera_app = CameraApp() if self.config["app_mode"] else None
        self.health = HealthReporter(self.camera_app.app, self.config) if self.config["app_mode"] else None
        self.network_handler = NetworkTableHandler(self.config["network_tables_ip"]) if self.config["use_network_tables"] else None

        if self.config["app_mode"]:
            threading.Thread(target=self.camera_app.run, daemon=True).start()
            if self.health and self.cameras:
                self.health.set_camera(self.cameras[0])  # Report fist camera for health
            if self.network_handler and self.health:
                self.health.set_network_handler(self.network_handler)

        if len(self.cameras) == 0:
            self.logger.warning("No cameras provided to VisionCore. Vision will not run.")
            self.camera_handler = None
        elif len(self.cameras) == 1:
            self.logger.info("Single camera provided to VisionCore.")
            self.camera_handler = None
        else:
            self.logger.info(f"{len(self.cameras)} cameras provided to VisionCore.")
            self.camera_handler = MultipleCameraHandler(self.cameras)

        self.planner = PathPlanner(self.config)
        self.fuel_tracker = FuelTracker(self.config)

        self.recorder = VideoRecorder(output_dir="VideoRecordings :)") if self.config["record_mode"] else None

    # These methods are really jsut to simpify stuff so it doesnt look as bad (like me :))
    def _record_metrics(self, **kwargs):
        if self.metrics:
            self.metrics.record(**kwargs)

    def _tick_metrics(self):
        if self.metrics:
            self.metrics.tick()

    def _destroy_metrics(self):
        if self.metrics:
            self.metrics.destroy()

    def numpy_to_fuel_list(self, fuel_positions: np.ndarray) -> list[Fuel]:
        return [Fuel(p[0], p[1]) for p in fuel_positions]

    def run_multi_vision(self, camera_handler: MultipleCameraHandler):
        try:
            raw_fuel_positions = camera_handler.predict()
            return self.numpy_to_fuel_list(raw_fuel_positions), camera_handler.get_combined_frame()
        except Exception as e:
            self.logger.exception(f"Vision exception: {e}")
            return [], None

    def run_solo_vision(self, camera: Camera):
        try:
            raw_fuel_positions, annotated_frame = camera.run()
            return self.numpy_to_fuel_list(raw_fuel_positions), annotated_frame
        except Exception as e:
            self.logger.exception(f"Vision exception: {e}")
            return [], None

    def run(self):
        if len(self.cameras) == 0:
            self.logger.error("No cameras provided to VisionCore.")
            return
        elif len(self.cameras) == 1:
            self.logger.info("Running in solo camera mode.")
            self.run_solo_mode()
        else:
            self.logger.info("Running in multi camera mode.")
            self.run_multi_mode()

    def run_solo_mode(self):
        camera = self.cameras[0]
        try:
            self.logger.info("Starting solo camera loop.")
            self.logger.info("Warming up...")

            self.run_solo_vision(camera)  # Warm up

            self.logger.info("Warmed up.")

            while not self.shutdown_event.is_set():
                start_time = time.perf_counter()

                camera_lag_s = camera.get_frame_age()

                vision_start = time.perf_counter()
                fuel_list, annotated_frame = self.run_solo_vision(camera)
                vision_s = time.perf_counter() - vision_start

                if self.network_handler:
                    robot_pose = self.network_handler.get_robot_pose()
                    fuel_list = self.fuel_tracker.update(
                        fuel_list,
                        robot_pose.X(),
                        robot_pose.Y(),
                        robot_pose.rotation().radians()
                    )
                else:
                    fuel_list = self.fuel_tracker.update(fuel_list, 0, 0, 0)

                flask_s = None
                if self.config["app_mode"]:
                    if annotated_frame is None:
                        self.logger.warning("Annotated frame not returned from camera.")
                    else:
                        flask_start = time.perf_counter()
                        self.camera_app.set_frame(annotated_frame)
                        flask_s = time.perf_counter() - flask_start

                if len(fuel_list) == 0:
                    self.logger.warning("No fuel positions detected. Skipping loop iteration.")
                    loop_s = time.perf_counter() - start_time
                    self._record_metrics(loop_s=loop_s, vision_s=vision_s, camera_lag_s=camera_lag_s, flask_s=flask_s)
                    self._tick_metrics()
                    print(f"\rFPS: {1/loop_s:.1f}      ", end="")
                    continue

                _, fuel_list = self.planner.update_fuel_positions(fuel_list)

                network_s = None
                if self.config["use_network_tables"]:
                    network_start = time.perf_counter()
                    loop_s = time.perf_counter() - start_time
                    self.network_handler.send_fuel_list(fuel_list, "vision_data", "VisionData")
                    self.network_handler.send_data(1 / loop_s if loop_s > 0 else 0, "fps", "VisionData")
                    self.network_handler.send_data(len(fuel_list), "num_detections", "VisionData")
                    self.network_handler.send_data(camera_lag_s, "camera_lag", "VisionData")

                    data = camera.get_data_for_subsystem("hopper")
                    if data is not None:
                        self.network_handler.send_boolean(data, "hopper_sees_object", "VisionData")

                    network_s = time.perf_counter() - network_start

                loop_s = time.perf_counter() - start_time

                health_s = None
                if self.health:
                    health_start = time.perf_counter()
                    self.health.tick(fps=1 / loop_s if loop_s > 0 else 0, vision_s=vision_s, detections=len(fuel_list))
                    health_s = time.perf_counter() - health_start

                self._record_metrics(
                    loop_s=loop_s,
                    vision_s=vision_s,
                    camera_lag_s=camera_lag_s,
                    flask_s=flask_s,
                    network_s=network_s,
                    health_s=health_s,
                )
                self._tick_metrics()

                self.logger.info(f"FPS: {1/loop_s:.1f}")
                print(f"\rFPS: {1/loop_s:.1f}      ", end="")
        finally:
            camera.destroy()
            self._destroy_metrics()

    def run_multi_mode(self):
        try:
            self.logger.info("Starting multi camera loop.")
            self.logger.info("Warming up...")

            self.run_multi_vision(self.camera_handler)  # Warm up

            self.logger.info("Warmed up.")

            while not self.shutdown_event.is_set():
                start_time = time.perf_counter()

                ages = [cam.get_frame_age() for cam in self.camera_handler.cameras]
                camera_lag_s = sum(ages) / len(ages) if ages else 0.0

                vision_start = time.perf_counter()
                fuel_list, combined_frame = self.run_multi_vision(self.camera_handler)
                vision_s = time.perf_counter() - vision_start

                if self.network_handler:
                    robot_pose = self.network_handler.get_robot_pose()
                    fuel_list = self.fuel_tracker.update(
                        fuel_list,
                        robot_pose.X(),
                        robot_pose.Y(),
                        robot_pose.rotation().radians()
                    )
                else:
                    fuel_list = self.fuel_tracker.update(fuel_list, 0, 0, 0)

                flask_s = None
                if self.config["app_mode"]:
                    if combined_frame is None:
                        self.logger.warning("Combined frame not returned from camera handler.")
                    else:
                        flask_start = time.perf_counter()
                        self.camera_app.set_frame(combined_frame)
                        flask_s = time.perf_counter() - flask_start

                if len(fuel_list) == 0:
                    self.logger.warning("No fuel positions detected. Skipping loop iteration.")
                    loop_s = time.perf_counter() - start_time
                    self._record_metrics(loop_s=loop_s, vision_s=vision_s, camera_lag_s=camera_lag_s, flask_s=flask_s)
                    self._tick_metrics()
                    print(f"\rFPS: {1/loop_s:.1f}      ", end="")
                    continue

                _, fuel_list = self.planner.update_fuel_positions(fuel_list)

                network_s = None
                if self.config["use_network_tables"]:
                    network_start = time.perf_counter()
                    loop_s = time.perf_counter() - start_time
                    self.network_handler.send_fuel_list(fuel_list, "vision_data", "VisionData")
                    self.network_handler.send_data(1 / loop_s if loop_s > 0 else 0, "fps", "VisionData")
                    self.network_handler.send_data(len(fuel_list), "num_detections", "VisionData")
                    self.network_handler.send_data(camera_lag_s, "camera_lag", "VisionData")

                    for camera in self.camera_handler.cameras:
                        data = camera.get_data_for_subsystem("hopper")
                        if data is not None:
                            self.network_handler.send_boolean(data, "hopper_sees_object", "VisionData")

                    network_s = time.perf_counter() - network_start

                loop_s = time.perf_counter() - start_time

                health_s = None
                if self.health:
                    health_start = time.perf_counter()
                    self.health.tick(fps=1 / loop_s if loop_s > 0 else 0, vision_s=vision_s, detections=len(fuel_list))
                    health_s = time.perf_counter() - health_start

                self._record_metrics(
                    loop_s=loop_s,
                    vision_s=vision_s,
                    camera_lag_s=camera_lag_s,
                    flask_s=flask_s,
                    network_s=network_s,
                    health_s=health_s,
                )
                self._tick_metrics()

                self.logger.info(f"FPS: {1/loop_s:.1f}")
                print(f"\rFPS: {1/loop_s:.1f}      ", end="")
        finally:
            self.camera_handler.destroy()
            self._destroy_metrics()