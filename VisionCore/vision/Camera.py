import cv2
import math
import numpy as np
from torch import device
from ultralytics import YOLO
import time
import os
import logging
import sys
from scipy.spatial.transform import Rotation
import threading
import queue
from .genericYolo import Box, Results, YoloWrapper
from rknnlite.api import RKNNLite  # No error handling :)
from VisionCore.config.VisionCoreConfig import VisionCoreConfig, VisionCoreCameraConfig
import subprocess

class Camera:
    """Simple VisionCore starter Camera, so I can later expand into AprilTags and stuff. All it does """
    def __init__(
        self,
        camera_config: VisionCoreConfig,
        config: VisionCoreCameraConfig
    ):
        self.logger = logging.getLogger(__name__)
        self.source = camera_config["source"]

        # Unit conversion dict
        self.conversions = {
            "meter": 0.0254,
            "meters": 0.0254,
            "inch": 1.0,
            "inches": 1.0,
            "foot": 1 / 12,
            "feet": 1 / 12,
            "centimeter": 2.54,
            "centimeters": 2.54,
        }

        # Yolo/RKNN vision stuff
        self.unit = config["unit"]
        self.debug_mode = config["debug_mode"]

        # self.logger.info(f"Camera object created with: {self.__dict__}")

        # Setups the buffering/timing stuff and some scripted values
        self.frame_timestamp = None
        self._last_result = None
        self._last_frame = None
        self._fresh_frame = False
        self.stopped = False
        self.frame = None
        self.gui_available = False  # Figure out how to dynamically figure this out
        self.last_time = time.perf_counter()
        self.frame_timeout = 1.0 / max(self.fps_cap, 1)

        # The stuff below this handles all the camera/photo buffering, timing stuff thats really complex and hard to explain
        if isinstance(self.source, str) and self.source.lower().endswith(
            (".png", ".jpg", ".jpeg", ".bmp")
        ):
            self.is_image = True
            self.image = cv2.imread(self.source)
            if self.image is None:
                raise ValueError(f"Failed to read image {self.source}")
        else:
            # Assume itss a video file or webcam index
            self.is_image = False
            device = self.source if isinstance(self.source, str) else f"/dev/video{self.source}"

            self.cap = cv2.VideoCapture(self.source, cv2.CAP_V4L2)

            if not self.cap.isOpened():
                raise ValueError("Camera failed to open")

            for _ in range(10):
                self.cap.grab()

            subprocess.run([
                "v4l2-ctl", "-d", device,
                "--set-fmt-video=width=320,height=320,pixelformat=MJPG"
            ], capture_output=True)

            time.sleep(0.15)

            self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 320)
            self.cap.set(cv2.CAP_PROP_FPS, self.fps_cap)

            for _ in range(20):
                self.cap.grab()

            fmt = self.cap.get(cv2.CAP_PROP_FOURCC)
            self.logger.info(f"FOURCC: {fmt}")

            if not self.cap.isOpened():
                self.logger.error(f"Cannot open source {self.source}")
                raise ValueError(f"Cannot open source {self.source}.")

        self.frame_lock = threading.Lock()
        self._frame_event = (
            threading.Event()
        )  # signals preproc worker that a new frame is ready

        self._preproc_q: queue.Queue = queue.Queue(maxsize=1)
        self._use_pipeline = (not self.is_image) and (self.model.model_type == "rknn")

        if not self.is_image:
            threading.Thread(target=self._reader, daemon=True).start()
            if self._use_pipeline:
                threading.Thread(target=self._preprocess_worker, daemon=True).start()

    def _reader(self):
        while not self.stopped:
            ret, frame = self.cap.read()
            if not ret:
                self.logger.warning(
                    f"Failed to retrieve frame from, attempting to continue: {self.source}"
                )
                time.sleep(0.05)
                continue

            # frame[:, :, 0], frame[:, :, 2] = frame[:, :, 2].copy(), frame[:, :, 0].copy() ONLY IF STILL PINK AFTER EVERYTHING ELSE, LAST RESORT YUYV

            # if np.mean(frame) < 1: // To computationally heavy
            if frame.max() < 1:
                self.logger.debug("Frame is a solid color, skipping...")
                continue

            with self.frame_lock:
                self.frame = frame
                self.frame_timestamp = time.perf_counter()
            self._frame_event.set()  # wake up preproc worker immediately instead of making it poll

            # self.frame = frame
            # time.sleep(0.002)  # Help not overuse CPU

    def _preprocess_worker(self):
        last_ts = None
        h, w = self.input_size[1], self.input_size[0]
        bufs = [
            np.empty((1, h, w, 3), dtype=np.uint8),
            np.empty((1, h, w, 3), dtype=np.uint8),
        ]
        buf_idx = 0

        while not self.stopped:
            with self.frame_lock:
                frame = self.frame
                ts = self.frame_timestamp

            if frame is None or ts == last_ts:
                self._frame_event.wait(timeout=0.05)
                self._frame_event.clear()
                continue

            if not self._preproc_q.empty():
                time.sleep(0.005)
                continue

            last_ts = ts
            orig_shape = frame.shape

            img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            if self.grayscale:
                gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
                img_rgb = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)

            self._letterbox_into(img_rgb, bufs[buf_idx][0], self.input_size)
            self._preproc_q.put((bufs[buf_idx], frame, orig_shape))
            buf_idx = 1 - buf_idx

    def get_frame_age(self) -> float | None:
        with self.frame_lock:
            ts = self.frame_timestamp
        if ts is None:
            return 0.0
        return time.perf_counter() - ts

    def get_frame(self, preprocessed=False):
        frame = None
        if self.is_image:
            frame = self.image.copy()
        else:
            with self.frame_lock:
                frame = self.frame.copy() if self.frame is not None else None

        if frame is None:
            return None

        if preprocessed:
            return self._preprocess_for_rknn(frame)

        return frame

    def run(self):
        data, frame = self.get_yolo_data()
        # self.logger.info(f"Got data: {data}, and frame: {frame}")
        if data is None or frame is None:
            self.logger.info("Frame or data is None")
            return np.empty((0, 2)), None

        img_h, img_w = frame.shape[:2]
        # self.logger.info(f"img_h: {img_h}, img_w: {img_w}")

        map_points = []

        # self.logger.info(f"Boxes: {data.boxes}")

        for box in data.boxes:
            # Filter boxes for things like confidence, apsect, etc.
            if not self._filter_box(box, img_w, img_h):
                continue

            # Transform from pixel coordinates to robot-relative coordinates
            pt = self._box_to_robot_point(box, img_w, img_h)
            if pt is not None:
                map_points.append(pt)
            else:
                self.logger.info("Skipping detection due to illegal shape.")

        return np.array(map_points) if map_points else np.empty((0, 2)), frame

    def get_subsystem(self):
        return self.subsystem

    def destroy(self):
        self.stopped = True
        if not self.is_image:
            self.cap.release()
        cv2.destroyAllWindows()

    def release(self):
        if not self.is_image and hasattr(self, "cap") and self.cap:
            self.cap.release()
