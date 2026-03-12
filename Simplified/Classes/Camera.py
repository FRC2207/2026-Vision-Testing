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
from .GenericYolo.genericYolo import Box, Results, YoloWrapper

class Camera:
    __slots__ = (
        'source', 'camera_fov', 'known_calibration_distance', 'ball_d_inches',
        'known_calibration_pixel_height', 'subsystem', 'margin', 'min_confidence',
        'grayscale', 'yolo_model_file', 'camera_pitch_angle', 'camera_height',
        'camera_x', 'camera_y', 'camera_bot_relative_yaw', 'debug_mode',
        'ball_count', 'gui_available', 'logger', 'last_time', 'cap',
        'focal_length_pixels', 'model', 'ret', 'frame', "__dict__"
    )

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

        # Camera transform stuff
        self.camera_pitch_angle = camera_downward_angle
        self.camera_height = camera_height
        self.camera_x = camera_x
        self.camera_y = camera_y
        self.camera_bot_relative_yaw = camera_bot_relative_angle

        self.debug_mode = debug_mode
        self.ball_count = 0

        self.gui_available = True
        self.logger = logging.getLogger(__name__)
        self.logger.info(f"Camera object created with: {self.__dict__}")

        self.last_time = time.perf_counter()

        self.cap = cv2.VideoCapture(self.source)
        self.stopped = False
        self.ret, self.frame = self.cap.read()
        # self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        # time.sleep(0.5)

        if not self.cap.isOpened():
            self.logger.error(f"Cannot open source {self.source}")
            raise ValueError(f"Cannot open source {self.source}")

        self.focal_length_pixels = (
            self.known_calibration_pixel_height * self.known_calibration_distance
        ) / self.ball_d_inches

        self.model = YoloWrapper(self.yolo_model_file)
        threading.Thread(target=self._reader, daemon=True).start()

    def _reader(self):
        while not self.stopped:
            ret, frame = self.cap.read()
            if not ret:
                self.logger.error(f"Failed to retrieve frame from: {self.source}")
                raise ValueError(f"Failed to retrieve frame from: {self.source}")
            if self.grayscale:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)  # Restore 3 channels, it wil still be gray tho

            self.frame = frame

    def get_frame(self):
        return self.frame

    def get_yolo_data(self):
        frame = self.get_frame()
        # If grayscale, convert back to 3 channels for YOLO model
        if self.grayscale:
            frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)

        results = self.model.predict(frame)

        annotated_frame = frame.copy()

        # Show it with cv2
        if self.debug_mode:
            annotated_frame = results.plot(frame)
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
            x1, y1, x2, y2 = box.xyxy
            w_pixels = x2 - x1
            h_pixels = y2 - y1
            conf = box.conf

            # Only accept things with a high enough confidence
            if conf < self.min_confidence:
                continue
            # Only accept boxes that are in the margin
            if x1 < self.margin or y1 < self.margin or x2 > (img_w - self.margin):
                continue
            aspect_ratio = w_pixels / h_pixels
            # Only accept boxes that are roughly square
            if not (0.8 <= aspect_ratio <= 1.2):
                continue

            cx = (x1 + x2) / 2.0
            cy = (y1 + y2) / 2.0


            avg_pixels = (w_pixels + h_pixels) / 2.0
            if avg_pixels <= 0:
                continue

            distance_los = (self.ball_d_inches * self.focal_length_pixels) / avg_pixels

            # Transform from pixel coordinates to robot-relative coordinates
            robot_point = self._pixel_to_robot_coordinates(
                cx, cy, distance_los, img_w, img_h
            )

            if robot_point is not None:
                map_points.append(robot_point)

        return np.array(map_points) if map_points else np.empty((0, 2))

    def run(self):
        data, frame = self.get_yolo_data()
        img_h, img_w = data.orig_shape[:2]

        map_points = []

        for box in data.boxes:
            x1, y1, x2, y2 = box.xyxy
            w_pixels = x2 - x1
            h_pixels = y2 - y1
            conf = box.conf

            # Only accept things with a high enough confidence
            if conf < self.min_confidence:
                continue
            # Only accept boxes that are in the margin
            if x1 < self.margin or y1 < self.margin or x2 > (img_w - self.margin):
                continue
            aspect_ratio = w_pixels / h_pixels
            # Only accept boxes that are roughly square
            if not (0.8 <= aspect_ratio <= 1.2):
                continue

            cx = (x1 + x2) / 2.0
            cy = (y1 + y2) / 2.0


            avg_pixels = (w_pixels + h_pixels) / 2.0
            if avg_pixels <= 0:
                continue

            distance_los = (self.ball_d_inches * self.focal_length_pixels) / avg_pixels

            # Transform from pixel coordinates to robot-relative coordinates
            robot_point = self._pixel_to_robot_coordinates(
                cx, cy, distance_los, img_w, img_h
            )

            if robot_point is not None:
                map_points.append(robot_point)

        return np.array(map_points) if map_points else np.empty((0, 2))

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
        # Its partly vibe coded but I reviewd it but also im failing my precalc class so idk

        pixel_offset_x = pixel_x - (img_w / 2.0)
        horizontal_angle_rad = math.atan(pixel_offset_x / self.focal_length_pixels)

        camera_height = self.camera_height

        if camera_height > 0 and distance_los > camera_height:
            true_horizontal_distance = math.sqrt(distance_los**2 - camera_height**2)
        else:
            true_horizontal_distance = distance_los

        left_right_distance = true_horizontal_distance * math.sin(horizontal_angle_rad)
        forward_distance = true_horizontal_distance * math.cos(horizontal_angle_rad)

        yaw_rad = math.radians(self.camera_bot_relative_yaw)
        cos_yaw = math.cos(yaw_rad)
        sin_yaw = math.sin(yaw_rad)

        x_rotated = forward_distance * sin_yaw + left_right_distance * cos_yaw
        y_rotated = forward_distance * cos_yaw - left_right_distance * sin_yaw

        x_inches = x_rotated + self.camera_x
        y_inches = y_rotated + self.camera_y

        x_meters = x_inches * 0.0254
        y_meters = y_inches * 0.0254

        return np.array([x_meters, y_meters], dtype=np.float32)

    def destroy(self):
        self.stopped = True
        self.cap.release()
        cv2.destroyAllWindows()