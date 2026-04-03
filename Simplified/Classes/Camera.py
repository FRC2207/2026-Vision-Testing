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
from .GenericYolo.genericYolo import Box, Results, YoloWrapper
from rknnlite.api import RKNNLite  # No error handling :)
from VisionCoreConfig import VisionCoreConfig, VisionCoreCameraConfig
import subprocess

class Camera:
    def __init__(
        self,
        camera_config: VisionCoreConfig,
        config: VisionCoreCameraConfig,
        core_mask=RKNNLite.NPU_CORE_0, # Eventually ill figure out how move into config
        # core_mask=None
    ):
        self.logger = logging.getLogger(__name__)
        self.source = camera_config["source"]

        # Camera calibration stuff
        try:
            self.known_calibration_distance = camera_config["calibration"]["distance"]
            self.ball_d_inches = camera_config["calibration"]["game_piece_size"]
            self.known_calibration_pixel_height = camera_config["calibration"]["size"]
            self.fov = camera_config["calibration"]["fov"]
            self.grayscale = True if camera_config["grayscale"] == True else False
            self.fps_cap = camera_config["fps_cap"]
            self.subsystem = camera_config["subsystem"]

            # Focal length calc's (short for calculations)
            if camera_config.get("focal_length_pixels") is None:
                try:
                    self.logger.info(
                        "No focal length in VisionCore config, calculating from calibration data..."
                    )
                    self.focal_length_pixels = (
                        self.known_calibration_pixel_height
                        * self.known_calibration_distance
                    ) / self.ball_d_inches
                    self.logger.info("Focal length calculated: {:.2f} pixels".format(self.focal_length_pixels))
                except ZeroDivisionError:
                    self.logger.warning(
                        "Calibration game piece size is zero, cannot calculate focal length. Defaulting to 1."
                    )
                    self.focal_length_pixels = 1.0
            else:
                self.focal_length_pixels = camera_config["focal_length_pixels"]

            # Camera transform stuff
            self.camera_bot_relative_yaw = camera_config["yaw"]
            self.camera_pitch_angle = camera_config["pitch"]
            self.camera_height = camera_config["height"]
            self.camera_x = camera_config["x"]
            self.camera_y = camera_config["y"]
        except KeyError as e:
            raise ValueError(f"Missing camera config key: {e}")

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
        self.margin = config["vision_model"].get("margin", 0)
        self.min_confidence = config["vision_model"].get("min_conf", 0.5)
        self.yolo_model_file = config["vision_model"]["file_path"]
        self.input_size = config["vision_model"]["input_size"]
        self.quantized = config["vision_model"].get("quantized", False)

        self.core_mask = core_mask
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

            for _ in range(20):
                self.cap.grab()

            fmt = self.cap.get(cv2.CAP_PROP_FOURCC)
            self.logger.info(f"FOURCC: {fmt}")

            if not self.cap.isOpened():
                self.logger.error(f"Cannot open source {self.source}")
                raise ValueError(f"Cannot open source {self.source}.")

        self.model = YoloWrapper(
            self.yolo_model_file,
            self.core_mask,
            self.input_size,
            quantized=self.quantized,
        )

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

    def _letterbox_into(self, img, dst, target_size):
        h, w = img.shape[:2]
        target_w, target_h = target_size
        scale = min(target_w / w, target_h / h)
        new_w = int(w * scale)
        new_h = int(h * scale)
        pad_w = target_w - new_w
        pad_h = target_h - new_h
        top = pad_h // 2
        left = pad_w // 2

        resized = cv2.resize(img, (new_w, new_h))
        dst[:] = 114
        dst[top : top + new_h, left : left + new_w] = resized

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

    def _letterbox(self, img, target_size=(640, 640)):
        h, w = img.shape[:2]
        target_w, target_h = target_size
        scale = min(target_w / w, target_h / h)
        new_w, new_h = int(w * scale), int(h * scale)
        resized = cv2.resize(img, (new_w, new_h))

        pad_w = target_w - new_w
        pad_h = target_h - new_h
        top = pad_h // 2
        bottom = pad_h - top
        left = pad_w // 2
        right = pad_w - left
        padded = cv2.copyMakeBorder(
            resized,
            top,
            bottom,
            left,
            right,
            cv2.BORDER_CONSTANT,
            value=(114, 114, 114),
        )
        return padded, scale, left, top

    def _preprocess_for_rknn(self, frame):
        if frame is None:
            return None

        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        img_resized, _, left, top = self._letterbox(img_rgb, self.input_size)

        img_input = np.expand_dims(img_resized, axis=0)  # (1, 640, 640, 3)
        img_input = np.ascontiguousarray(img_input, dtype=np.uint8)
        return img_input

    def _filter_box(self, box: Box, img_w: int, img_h: int) -> bool:
        x1, y1, x2, y2 = box.xyxy
        w_px = x2 - x1
        h_px = y2 - y1

        if box.conf < self.min_confidence:
            # self.logger.info("Skipping detection due to low confidence.")
            return False
        # if x1 < self.margin or y1 < self.margin or x2 > (img_w - self.margin):
        if x1 < self.margin or y1 < self.margin or x2 > (img_w - self.margin) or y2 > (img_h - self.margin):
            # self.logger.info("Skipping detection due to margin.")
            return False
        if h_px == 0:
            return False
        aspect_ratio = w_px / h_px
        if not (0.8 <= aspect_ratio <= 1.2):
            # self.logger.info("Skipping detection due to rectangular shape.")
            return False
        return True

    def _box_to_robot_point(
        self, box: Box, img_w: int, img_h: int
    ) -> np.ndarray | None:
        x1, y1, x2, y2 = box.xyxy
        w_px = x2 - x1
        h_px = y2 - y1
        avg_pixels = (w_px + h_px) / 2.0
        if avg_pixels <= 0:
            return None
        cx = (x1 + x2) / 2.0
        cy = (y1 + y2) / 2.0
        distance_los = (self.ball_d_inches * self.focal_length_pixels) / avg_pixels
        return self._pixel_to_robot_coordinates(cx, cy, distance_los, img_w, img_h)

    def get_yolo_data(self):
        if self._use_pipeline:
            try:
                preprocessed, orig_frame, orig_shape = self._preproc_q.get(timeout=self.frame_timeout)
            except queue.Empty:
                if self._last_result is not None:
                    self._fresh_frame = False
                    return self._last_result, self._last_frame
                return None, None

            results = self.model.predict_preprocessed(preprocessed, orig_shape)
            annotated_frame = orig_frame
            self._last_result = results
            self._last_frame = annotated_frame
            self._fresh_frame = True
        else:
            # self.logger.info("Calling: self.get_frame()")
            frame = self.get_frame(preprocessed=False)

            if frame is None:
                self.logger.warning(
                    "Frame not retrieved properly from camera (frame was None)"
                )
                return None, None

            results = self.model.predict(frame, orig_shape=frame.shape)
            annotated_frame = frame.copy()
            self._fresh_frame = True

        # Show it with cv2
        if self.debug_mode and self._fresh_frame:
            self.logger.info("Plotting frame")
            annotated_frame = results.plot(annotated_frame.copy())
            new_time_frame = time.perf_counter()
            fps = 1 / (new_time_frame - self.last_time)
            self.last_time = new_time_frame

            # Format and display FPS on the frame
            fps_text = f"FPS: {int(fps)}"
            cv2.putText(
                annotated_frame,
                fps_text,
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 0, 255),
                2,
                cv2.LINE_AA,
            )
            self._last_frame = annotated_frame
            if self.gui_available:
                cv2.imshow("YOLO Detections", annotated_frame)
                cv2.waitKey(1)

        return results, annotated_frame

    def run_with_supplied_data(self, data):
        img_h, img_w = data.orig_shape[:2]

        map_points = []

        for box in data.boxes:
            if not self._filter_box(box, img_w, img_h):
                continue

            # Transform from pixel coordinates to robot-relative coordinates
            pt = self._box_to_robot_point(box, img_w, img_h)
            if pt is not None:
                map_points.append(pt)
        return np.array(map_points) if map_points else np.empty((0, 2))
    
    def get_data_for_subsytem(self, target: str):
        if self.subsystem == target:
            if self.subsystem == "field":
                return self.run()
            elif self.subsystem == "hopper":
                return True if self.run()[0].shape[0] > 0 else False
            else:
                self.logger.warning(f"Unknown subsystem: {self.subsystem}, defaulting to field data.")
                return self.run()
        else:
            return None

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

    def _pixel_to_robot_coordinates(self, pixel_x, pixel_y, distance_los, img_w, img_h):
        pixel_offset_x = pixel_x - (img_w / 2.0)
        horizontal_angle_rad = math.atan(pixel_offset_x / self.focal_length_pixels)

        # distance_los is true 3D distance from apparent size — just project to ground
        if self.camera_height > 0 and distance_los > self.camera_height:
            true_horizontal_distance = math.sqrt(distance_los**2 - self.camera_height**2)
        else:
            true_horizontal_distance = distance_los * math.cos(math.radians(self.camera_pitch_angle))

        left_right_distance = true_horizontal_distance * math.sin(horizontal_angle_rad)
        forward_distance    = true_horizontal_distance * math.cos(horizontal_angle_rad)

        yaw_rad = math.radians(self.camera_bot_relative_yaw)
        cos_yaw = math.cos(yaw_rad)
        sin_yaw = math.sin(yaw_rad)

        x_rotated = forward_distance * cos_yaw + left_right_distance * sin_yaw
        y_rotated = forward_distance * sin_yaw - left_right_distance * cos_yaw

        if self.unit not in self.conversions:
            self.unit = "meter"
        scale = self.conversions[self.unit]

        x_out = (x_rotated + self.camera_x) * scale
        y_out = (y_rotated + self.camera_y) * scale

        return np.array([x_out, y_out], dtype=np.float32)

    def destroy(self):
        self.stopped = True
        if not self.is_image:
            self.cap.release()
        cv2.destroyAllWindows()

    def release(self):
        if not self.is_image and hasattr(self, "cap") and self.cap:
            self.cap.release()
