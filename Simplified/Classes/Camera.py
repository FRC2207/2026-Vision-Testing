import cv2
import math
import numpy as np
from ultralytics import YOLO
import time
import os
import logging
import sys
from scipy.spatial.transform import Rotation

try:
    from rknn.api import RKNN
except ImportError:
    RKNN = None


class Box:
    def __init__(self, xyxy, conf):
        self.xyxy = xyxy
        self.conf = conf


class Results:
    def __init__(self, boxes: list[Box], orig_shape):
        self.boxes = boxes
        self.orig_shape = orig_shape

    def plot(self, frame):
        # Custom plotting thingie since rknn doesnt have built in plottin like ultralytics
        for box in self.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        return frame


class YoloWrapper:
    def __init__(self, model_file: str, input_size=(640, 640)):
        self.model_file = model_file
        self.input_size = input_size
        self.model_type = None

        self.logger = logging.getLogger(__name__)

        if model_file.endswith(".rknn"):
            if RKNN is None:
                self.logger.error(
                    "Could node import RKNNLite. This could be because you meant to run a .pt or .onnx on a laptop, but if its the pi ur cooked."
                )
                raise ImportError(
                    "Could node import RKNNLite. This could be because you meant to run a .pt or .onnx on a laptop, but if its the pi ur cooked."
                )
            self.model_type = "rknn"
            self.model = RKNN()
            ret = self.model.load_rknn(self.model_file)
            if ret != 0:
                self.logger.error(f"Failed to load RKNN model: {self.model_file}")
                raise ValueError(f"Failed to load RKNN model: {self.model_file}")

            # Already built if .rknn file so skip
            # ret = self.model.build(do_quantization=False)

            # if ret != 0:
            #     raise ValueError(f"Failed to build RKNN model: {self.model_file}")

            ret = self.model.init_runtime(target="rk3588")
            if ret != 0:
                self.logger.error(
                    f"Failed to initialize RKNN runtime for model: {self.model_file}"
                )
                raise ValueError(
                    f"Failed to initialize RKNN runtime for model: {self.model_file}"
                )
        elif model_file.endswith(".onnx") or model_file.endswith(".pt"):
            self.model_type = "yolo"
            self.model = YOLO(self.model_file, verbose=False, task="detect")
        else:
            self.logger.error(
                f"Unsupported model file type: {self.model_file}. Check constants and spelling blud."
            )
            raise ValueError(
                f"Unsupported model file type: {self.model_file}. Check constants and spelling blud."
            )

    def predict(self, frame) -> Results:
        if self.model_type == "rknn":
            img = cv2.resize(frame, self.input_size)
            img = img.astype(np.float32) / 255.0
            img = np.expand_dims(img, axis=0)
            outputs = self.model.inference(inputs=[img])[0]

            return self._convert_rknn_outputs(outputs, frame.shape)
        elif self.model_type == "yolo":
            results = self.model(frame, verbose=False)[0]

            return self._convert_ultralytics_to_results(results)
        else:
            self.logger.error(
                f"Unsupported model type: {self.model_type}. This should never happen, you broke the matrix."
            )
            raise ValueError(
                f"Unsupported model type: {self.model_type}. This should never happen, you broke the matrix."
            )

    def _convert_ultralytics_to_results(self, results):
        boxes = [
            Box(list(map(float, box[:4])), float(conf))
            for box, conf in zip(results.boxes.xyxy, results.boxes.conf)
        ]
        return Results(boxes, results.orig_shape)

    def _convert_rknn_outputs(self, outputs, orig_shape):
        # I think the _ means like private method but im guessing at this point
        # Hopefully this will make it so that I don't have to chane the rest of the code to work with wtv rknn files output
        outputs = np.array(outputs)
        if (
            len(outputs.shape) == 3
        ):  # Prolly (1, N, 6) where N is number of detections and 6 is x1, y1, x2, y2, conf, class. I hate array dimension stuff so much
            outputs = outputs[0]

        boxes = [Box(list(map(float, box[:4])), float(box[4])) for box in outputs]

        return Results(boxes, orig_shape)

    def release(self):
        if self.model_type == "rknn":
            self.model.release()


class Camera:
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
        # self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        # time.sleep(0.5)

        if not self.cap.isOpened():
            self.logger.error(f"Cannot open source {self.source}")
            raise ValueError(f"Cannot open source {self.source}")

        self.focal_length_pixels = (
            self.known_calibration_pixel_height * self.known_calibration_distance
        ) / self.ball_d_inches

        self.model = YoloWrapper(self.yolo_model_file)

    def get_frame(self):
        ret, frame = self.cap.read()
        if not ret:
            self.logger.erro(f"Failed to retrieve frame from: {self.source}")
            raise ValueError(f"Failed to retrieve frame from: {self.source}")
        if self.grayscale:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        return frame

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

            if self.gui_available:  # Fixes a really anooying error I had
                cv2.imshow("YOLO Detections", annotated_frame)
                cv2.waitKey(1)

        return results, annotated_frame

    def run(self):
        data, frame = self.get_yolo_data()
        img_h, img_w = data.orig_shape[:2]

        map_points = []
        confidence_list = []

        for box in data.boxes:
            x1, y1, x2, y2 = box.xyxy
            conf = box.conf

            if conf < self.min_confidence:
                continue
            if x1 < self.margin or y1 < self.margin or x2 > (img_w - self.margin):
                continue

            confidence_list.append(conf)

            cx = (x1 + x2) / 2.0
            cy = (y1 + y2) / 2.0

            w_pixels = x2 - x1
            h_pixels = y2 - y1
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
        horizontal_angle_rad = math.radians(self.camera_fov * pixel_offset_x / img_w)

        pitch_rad = math.radians(self.camera_pitch_angle)

        camera_height = self.camera_height

        if camera_height > 0:
            true_horizontal_distance = math.sqrt(
                max(distance_los**2 - camera_height**2, 0.1)
            )
        else:
            true_horizontal_distance = distance_los

        aspect_ratio = img_h / img_w
        vertical_fov = self.camera_fov * aspect_ratio
        pixel_offset_y = pixel_y - (img_h / 2.0)
        angle_in_image_rad = math.radians(vertical_fov * pixel_offset_y / img_h)

        total_angle_to_object = pitch_rad + angle_in_image_rad

        actual_angle_to_object = math.atan(camera_height / true_horizontal_distance)

        left_right_distance = true_horizontal_distance * math.sin(horizontal_angle_rad)
        forward_distance = true_horizontal_distance * math.cos(horizontal_angle_rad)

        yaw_rad = math.radians(self.camera_bot_relative_yaw)

        cos_yaw = math.cos(yaw_rad)
        sin_yaw = math.sin(yaw_rad)

        x_rotated = forward_distance * sin_yaw + left_right_distance * cos_yaw
        y_rotated = forward_distance * cos_yaw - left_right_distance * sin_yaw

        x_inches = x_rotated + self.camera_x
        y_inches = y_rotated + self.camera_y

        return np.array([x_inches, y_inches], dtype=np.float32)

    def destroy(self):
        self.cap.release()
        cv2.destroyAllWindows()
