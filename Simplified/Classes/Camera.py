import cv2
import math
import numpy as np
from ultralytics import YOLO
import time
import os
import logging
import sys
from scipy.spatial.transform import Rotation
import threading
import queue
from .GenericYolo.genericYolo import Box, Results, YoloWrapper

# Consider adding skipping stale frames

class Camera:
    # __slots__ = (
    #     'source', 'camera_fov', 'known_calibration_distance', 'ball_d_inches',
    #     'known_calibration_pixel_height', 'subsystem', 'margin', 'min_confidence',
    #     'grayscale', 'yolo_model_file', 'camera_pitch_angle', 'camera_height',
    #     'camera_x', 'camera_y', 'camera_bot_relative_yaw', 'debug_mode',
    #     'ball_count', 'gui_available', 'logger', 'last_time', 'cap',
    #     'focal_length_pixels', 'model', 'ret', 'frame',
    #     'frame_lock', 'stopped',
    # )

    def __init__(
        self,
        source: int | str,
        camera_fov: int,
        known_calibration_distance: int,
        ball_d_inches: int,
        known_calibration_pixel_height: int,
        yolo_model_file: str,
        camera_downward_angle: float,
        camera_bot_relative_angle: float,
        camera_height: int,
        camera_x: int,
        camera_y: int,
        grayscale: bool = True,
        margin: int = 10,
        min_confidence: float = 0.5,
        debug_mode: bool = False,
        subsystem: str = "field",
        input_size: tuple[int, int]=(640, 640),
        quantized: bool=True,
        unit: str="inch",
        core_mask=RKNNLite.NPU_CORE_0_1_2,
        fps_cap: int=50
    ):
        self.source = source
        self.camera_fov = camera_fov
        self.known_calibration_distance = known_calibration_distance
        self.ball_d_inches = ball_d_inches
        self.known_calibration_pixel_height = known_calibration_pixel_height
        self.subsystem = subsystem
        self.margin = margin
        self.min_confidence = min_confidence
        self.grayscale = grayscale
        self.yolo_model_file = yolo_model_file
        self.input_size = input_size
        self.fps_cap = fps_cap

        self.quantized = quantized
        self.unit = unit

        # Camera transform stuff
        self.camera_pitch_angle = camera_downward_angle
        self.camera_height = camera_height
        self.camera_x = camera_x
        self.camera_y = camera_y
        self.camera_bot_relative_yaw = camera_bot_relative_angle

        self.debug_mode = debug_mode
        self.ball_count = 0

        self.core_mask = core_mask

        self.gui_available = False
        self.logger = logging.getLogger(__name__)
        self.logger.info(f"Camera object created with: {self.__dict__}")

        self.frame_timestamp = None

        self.conversions = {
            "meter": 0.0254,
            "inch":  1.0,
            "foot":  1/12,
            "cm":    2.54,
        }

        self.last_time = time.perf_counter()

        self.frame = None

        if isinstance(source, str) and source.lower().endswith((".png", ".jpg", ".jpeg", ".bmp")):
            self.is_image = True
            self.image = cv2.imread(source)
            if self.image is None:
                raise ValueError(f"Failed to read image {source}")
        else:
            # Assume it's a video file or webcam index
            self.is_image = False
            self.cap = cv2.VideoCapture(source)
            self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'))
            # self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

            if not self.cap.isOpened():
                self.logger.error(f"Cannot open source {self.source}")
                raise ValueError(f"Cannot open source {source}.")
            
            self.cap.set(cv2.CAP_PROP_FPS, self.fps_cap)

        self.stopped = False
        # self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        # self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        # time.sleep(0.5)

        self.focal_length_pixels = (
            self.known_calibration_pixel_height * self.known_calibration_distance
        ) / self.ball_d_inches

        self.model = YoloWrapper(self.yolo_model_file, self.core_mask, self.input_size, quantized=self.quantized)
        
        self.frame_lock = threading.Lock()

        # Preprocessing pipeline queue — holds (preprocessed_buf, orig_frame, orig_shape)
        # maxsize=1 so if inference is slow, the old stale frame gets evicted and
        # replaced with the latest one rather than building up a backlog
        self._preproc_q: queue.Queue = queue.Queue(maxsize=1)
        self._use_pipeline = (not self.is_image) and (self.model.model_type == "rknn")

        if not self.is_image:
            threading.Thread(target=self._reader, daemon=True).start()
            if self._use_pipeline:
                threading.Thread(target=self._preprocess_worker, daemon=True).start()

    def _reader(self):
        # self.logger.info(f"self.stopped: {self.stopped}")
        while not self.stopped:
            # self.logger.debug("Attempting to grab frame.")
            ret, frame = self.cap.read()
            # self.logger.debug(f"Frame grabbed: {frame}")
            if not ret:
                self.logger.warning(f"Failed to retrieve frame from, attempting to continue: {self.source}")
                # raise ValueError(f"Failed to retrieve frame from: {self.source}")
                time.sleep(0.05)
                continue
            if self.grayscale:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)  # Restore 3 channels, it wil still be gray tho

            # if np.mean(frame) < 1: // To computationally heavy 
            if frame.max() < 1:
                self.logger.debug("Frame is a solid color, skipping...")
                continue

            with self.frame_lock:
                self.frame = frame
                self.frame_timestamp = time.perf_counter()
                
            # self.frame = frame
            # time.sleep(0.01) # Help not overuse CPU

    def _preprocess_worker(self):
        last_ts = None

        h, w = self.input_size[1], self.input_size[0]
        bufs = [np.empty((1, h, w, 3), dtype=np.uint8), np.empty((1, h, w, 3), dtype=np.uint8)]
        buf_idx = 0

        while not self.stopped:
            with self.frame_lock:
                frame = self.frame
                ts    = self.frame_timestamp

            if frame is None or ts == last_ts:
                # No new frame yet
                time.sleep(0.0005)
                continue

            last_ts    = ts
            if not self._preproc_q.empty():
                continue

            # No frame.copy() needed — _reader replaces self.frame with a new object
            # each write, so our local reference stays valid after releasing the lock
            orig_shape = frame.shape

            # Preprocess into current buffer
            img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            self._letterbox_into(img_rgb, bufs[buf_idx][0], self.input_size)

            self._preproc_q.put((bufs[buf_idx], frame, orig_shape))
            buf_idx = 1 - buf_idx

    def _letterbox_into(self, img, dst, target_size):
        # Letterbox img into dst in-place, no allocation
        h, w = img.shape[:2]
        target_w, target_h = target_size
        scale  = min(target_w / w, target_h / h)
        new_w  = int(w * scale)
        new_h  = int(h * scale)
        pad_w  = target_w - new_w
        pad_h  = target_h - new_h
        top    = pad_h // 2
        left   = pad_w // 2

        resized = cv2.resize(img, (new_w, new_h))
        dst[:] = 114
        dst[top:top + new_h, left:left + new_w] = resized

    def get_frame_age(self) -> float | None:
        with self.frame_lock:
            ts = self.frame_timestamp
        if ts is None:
            return None
        return time.perf_counter() - ts

    def get_frame(self, preprocessed=False):
        frame = None
        if self.is_image:
            frame = self.image.copy()
        else:
            with self.frame_lock:
                frame = self.frame.copy()
        
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
        padded = cv2.copyMakeBorder(resized, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(114,114,114))
        return padded, scale, left, top

    def _preprocess_for_rknn(self, frame):
        if frame is None:
            return None

        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        img_resized, _, left, top = self._letterbox(img_rgb, self.input_size)

        img_input = np.expand_dims(img_resized, axis=0) # (1, 640, 640, 3)
        img_input = np.ascontiguousarray(img_input, dtype=np.uint8)
        return img_input

    def _filter_box(self, box: Box, img_w: int, img_h: int) -> bool:
        x1, y1, x2, y2 = box.xyxy
        w_px = x2 - x1
        h_px = y2 - y1

        if box.conf < self.min_confidence:
            # self.logger.info("Skipping detection due to low confidence.")
            return False
        if x1 < self.margin or y1 < self.margin or x2 > (img_w - self.margin):
            # self.logger.info("Skipping detection due to margin.")
            return False
        if h_px == 0:
            return False
        aspect_ratio = w_px / h_px
        if not (0.8 <= aspect_ratio <= 1.2):
            # self.logger.info("Skipping detection due to rectangular shape.")
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
            # Fast path: preproc thread already did the cvtColor + letterbox
            # concurrently with the last inference, so we just grab and go
            try:
                preprocessed, orig_frame, orig_shape = self._preproc_q.get(timeout=0.15)
            except queue.Empty:
                self.logger.warning("Preproc pipeline timed out — no frame available.")
                return None, None

            results       = self.model.predict_preprocessed(preprocessed, orig_shape)
            annotated_frame = orig_frame
        else:
            # self.logger.info("Calling: self.get_frame()")
            frame = self.get_frame(preprocessed=False)

            if frame is None:
                self.logger.warning("Frame not retrieved properly from camera (frame was None)")
                return None, None

            results = self.model.predict(frame, orig_shape=frame.shape)
            annotated_frame = frame.copy()

        # Show it with cv2
        if self.debug_mode:
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

    def calculate_from_mosaic(self, local_x, local_y, local_w, local_h):
        img_w = self.cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 640
        img_h = self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 480
        
        scale_x = img_w / 320.0
        scale_y = img_h / 320.0
        
        native_cx = local_x * scale_x
        native_cy = local_y * scale_y
        native_avg_px = ((local_w * scale_x) + (local_h * scale_y)) / 2.0
        
        if native_avg_px <= 0:
            return None

        distance_los = (self.ball_d_inches * self.focal_length_pixels) / native_avg_px

        return self._pixel_to_robot_coordinates(
            native_cx, native_cy, distance_los, int(img_w), int(img_h)
        )

    def get_subsystem(self):
        return self.subsystem

    def _pixel_to_robot_coordinates(
        self,
        pixel_x: float,
        pixel_y: float,
        distance_los: float,
        img_w: int,
        img_h: int,
    ) -> np.ndarray | None:
        # I have no clue if this math is actually right, but the tests say yes
        # Its partly vibe coded but I reviewed it but also im failing my precalc class so idk

        pixel_offset_x      = pixel_x - (img_w / 2.0)
        horizontal_angle_rad = math.atan(pixel_offset_x / self.focal_length_pixels)

        if self.camera_height > 0 and distance_los > self.camera_height:
            true_horizontal_distance = math.sqrt(distance_los**2 - self.camera_height**2)
        else:
            true_horizontal_distance = distance_los

        left_right_distance = true_horizontal_distance * math.sin(horizontal_angle_rad)
        forward_distance    = true_horizontal_distance * math.cos(horizontal_angle_rad)

        yaw_rad = math.radians(self.camera_bot_relative_yaw)
        cos_yaw = math.cos(yaw_rad)
        sin_yaw = math.sin(yaw_rad)

        x_rotated = forward_distance * sin_yaw + left_right_distance * cos_yaw
        y_rotated = forward_distance * cos_yaw - left_right_distance * sin_yaw

        scale = self.conversions.get(self.unit, 0.0254)
        if scale is None:
            self.logger.warning(f"Unknown unit: {self.unit}. Expected: {list(self.conversions.keys())}")
            self.unit = "inch"

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