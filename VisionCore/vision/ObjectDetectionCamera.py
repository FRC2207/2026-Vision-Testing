import cv2
import math
import numpy as np
import time
import logging
import threading
import queue
from .genericYolo import Box, Results, YoloWrapper
from VisionCore.vision.Camera import Camera


class ObjectDetectionCamera(Camera):
    def __init__(
        self,
        camera_config,
        config,
        core_mask=None,
    ):
        # First initialize base camera for frame acquisition
        super().__init__(camera_config, config)
        
        self.logger = logging.getLogger(__name__)

        try:
            self.known_calibration_distance = camera_config["calibration"]["distance"]
            self.ball_d_inches = camera_config["calibration"]["game_piece_size"]
            self.known_calibration_pixel_height = camera_config["calibration"]["size"]
            self.fov = camera_config["calibration"]["fov"]
            self.grayscale = camera_config.get("grayscale", False)
            self.fps_cap = camera_config.get("fps_cap", 30)
            self.subsystem = camera_config["subsystem"]

            # Calculate focal length from calibration or use provided value
            if camera_config.get("focal_length_pixels") is None:
                try:
                    self.logger.info("Calculating focal length from calibration data...")
                    self.focal_length_pixels = (
                        self.known_calibration_pixel_height
                        * self.known_calibration_distance
                    ) / self.ball_d_inches
                    self.logger.info(f"Focal length calculated: {self.focal_length_pixels:.2f} pixels")
                except ZeroDivisionError:
                    self.logger.warning("Calibration game piece size is zero, defaulting focal length to 1")
                    self.focal_length_pixels = 1.0
            else:
                self.focal_length_pixels = camera_config["focal_length_pixels"]

            # Camera pose relative to robot
            self.camera_bot_relative_yaw = camera_config["yaw"]
            self.camera_pitch_angle = camera_config["pitch"]
            self.camera_height = camera_config["height"]
            self.camera_x = camera_config["x"]
            self.camera_y = camera_config["y"]
        except KeyError as e:
            raise ValueError(f"Missing camera config key: {e}")

        # Unit conversion
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

        self.margin = config["vision_model"].get("margin", 0)
        self.min_confidence = config["vision_model"].get("min_conf", 0.5)
        self.yolo_model_file = config["vision_model"]["file_path"]
        self.input_size = config["vision_model"]["input_size"]
        self.quantized = config["vision_model"].get("quantized", False)
        self.core_mask = core_mask

        # Context/runtime settings
        self.unit = config["unit"]
        self.debug_mode = config["debug_mode"]
        self.gui_available = False

        self._last_result = None
        self._last_frame = None
        self._fresh_frame = False
        self.last_time = time.perf_counter()
        self.frame_timeout = 1.0 / max(self.fps_cap, 1)

        self.model = YoloWrapper(
            self.yolo_model_file,
            self.core_mask,
            self.input_size,
            quantized=self.quantized,
        )

        self._preproc_q = queue.Queue(maxsize=1)
        self._use_pipeline = (not self.is_image) and (self.model.model_type == "rknn")

        if not self.is_image and self._use_pipeline:
            threading.Thread(target=self._preprocess_worker, daemon=True).start()

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
        img_input = np.expand_dims(img_resized, axis=0)  # (1, H, W, 3)
        img_input = np.ascontiguousarray(img_input, dtype=np.uint8)
        return img_input

    def _filter_box(self, box: Box, img_w: int, img_h: int) -> bool:
        x1, y1, x2, y2 = box.xyxy
        w_px = x2 - x1
        h_px = y2 - y1

        if box.conf < self.min_confidence:
            return False
        if x1 < self.margin or y1 < self.margin or x2 > (img_w - self.margin) or y2 > (img_h - self.margin):
            return False
        if h_px == 0:
            return False
        aspect_ratio = w_px / h_px
        if not (0.8 <= aspect_ratio <= 1.2):
            return False
        return True

    def _box_to_robot_point(self, box: Box, img_w: int, img_h: int) -> np.ndarray | None:
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
            # Direct inference (CPU or non-RKNN GPU)
            frame = self.get_frame()
            if frame is None:
                self.logger.warning("Frame not retrieved from camera")
                return None, None

            results = self.model.predict(frame, orig_shape=frame.shape)
            annotated_frame = frame.copy()
            self._fresh_frame = True

        # Visualization in debug mode
        if self.debug_mode and self._fresh_frame:
            self.logger.info("Plotting detections")
            annotated_frame = results.plot(annotated_frame.copy())
            new_time = time.perf_counter()
            fps = 1 / (new_time - self.last_time) if (new_time - self.last_time) > 0 else 0
            self.last_time = new_time

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

            pt = self._box_to_robot_point(box, img_w, img_h)
            if pt is not None:
                map_points.append(pt)
        return np.array(map_points) if map_points else np.empty((0, 2))

    def run(self):
        data, frame = self.get_yolo_data()
        if data is None or frame is None:
            self.logger.info("Frame or data is None")
            return np.empty((0, 2)), None

        img_h, img_w = frame.shape[:2]
        map_points = []

        for box in data.boxes:
            if not self._filter_box(box, img_w, img_h):
                continue

            pt = self._box_to_robot_point(box, img_w, img_h)
            if pt is not None:
                map_points.append(pt)
            else:
                self.logger.info("Skipping detection due to illegal shape")

        return np.array(map_points) if map_points else np.empty((0, 2)), frame

    def get_data_for_subsystem(self, target: str):
        if self.subsystem == target:
            if self.subsystem == "field":
                return self.run()
            elif self.subsystem == "hopper":
                return True if self.run()[0].shape[0] > 0 else False
            else:
                self.logger.warning(f"Unknown subsystem: {self.subsystem}, defaulting to field data")
                return self.run()
        else:
            return None

    def _pixel_to_robot_coordinates(self, pixel_x, pixel_y, distance_los, img_w, img_h):
        pixel_offset_x = pixel_x - (img_w / 2.0)
        horizontal_angle_rad = math.atan(pixel_offset_x / self.focal_length_pixels)

        # Project line-of-sight distance to ground plane
        if self.camera_height > 0 and distance_los > self.camera_height:
            true_horizontal_distance = math.sqrt(distance_los**2 - self.camera_height**2)
        else:
            true_horizontal_distance = distance_los * math.cos(math.radians(self.camera_pitch_angle))

        left_right_distance = true_horizontal_distance * math.sin(horizontal_angle_rad)
        forward_distance = true_horizontal_distance * math.cos(horizontal_angle_rad)

        # Rotate by camera yaw
        yaw_rad = math.radians(self.camera_bot_relative_yaw)
        cos_yaw = math.cos(yaw_rad)
        sin_yaw = math.sin(yaw_rad)

        x_rotated = forward_distance * cos_yaw + left_right_distance * sin_yaw
        y_rotated = forward_distance * sin_yaw - left_right_distance * cos_yaw

        # Apply unit conversion and camera offset
        if self.unit not in self.conversions:
            self.unit = "meter"
        scale = self.conversions[self.unit]

        x_out = (x_rotated + self.camera_x) * scale
        y_out = (y_rotated + self.camera_y) * scale

        return np.array([x_out, y_out], dtype=np.float32)

    def get_subsystem(self):
        return self.subsystem

    def destroy(self):
        self.stopped = True
        if not self.is_image and hasattr(self, "cap") and self.cap:
            self.cap.release()
        cv2.destroyAllWindows()

    def release(self):
        if not self.is_image and hasattr(self, "cap") and self.cap:
            self.cap.release()