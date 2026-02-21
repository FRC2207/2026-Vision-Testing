import cv2
import math
import numpy as np
from ultralytics import YOLO
import time
import os
import logging
import sys

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
                self.logger.error("Could node import RKNNLite. This could be because you meant to run a .pt or .onnx on a laptop, but if its the pi ur cooked.")
                raise ImportError("Could node import RKNNLite. This could be because you meant to run a .pt or .onnx on a laptop, but if its the pi ur cooked.")
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
                self.logger.error(f"Failed to initialize RKNN runtime for model: {self.model_file}")
                raise ValueError(f"Failed to initialize RKNN runtime for model: {self.model_file}")
        elif model_file.endswith(".onnx") or model_file.endswith(".pt"):
            self.model_type = "yolo"
            self.model = YOLO(self.model_file, verbose=False, task="detect")
        else:
            self.logger.error(f"Unsupported model file type: {self.model_file}. Check constants and spelling blud.")
            raise ValueError(f"Unsupported model file type: {self.model_file}. Check constants and spelling blud.")
        
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
            self.logger.error(f"Unsupported model type: {self.model_type}. This should never happen, you broke the matrix.")
            raise ValueError(f"Unsupported model type: {self.model_type}. This should never happen, you broke the matrix.")
    
    def _convert_ultralytics_to_results(self, results):
        boxes = [Box(list(map(float, box[:4])), float(conf)) 
                 for box, conf in zip(results.boxes.xyxy, results.boxes.conf)]
        return Results(boxes, results.orig_shape)

    def _convert_rknn_outputs(self, outputs, orig_shape):
        # I think the _ means like private method but im guessing at this point
        # Hopefully this will make it so that I don't have to chane the rest of the code to work with wtv rknn files output
        outputs = np.array(outputs)
        if len(outputs.shape) == 3: # Prolly (1, N, 6) where N is number of detections and 6 is x1, y1, x2, y2, conf, class. I hate array dimension stuff so much
            outputs = outputs[0]

        boxes = [Box(list(map(float, box[:4])), float(box[4])) for box in outputs]
        
        return Results(boxes, orig_shape)

    def release(self):
        if self.model_type == "rknn":
            self.model.release()

class Camera:
    def __init__(self, source: int|str, camera_fov: int, known_calibration_distance: int, ball_d_inches: int, known_calibration_pixel_height: int, yolo_model_file: str,
                 camera_downward_angle: float, camera_bot_relative_angle: float ,camera_height: int, camera_x: int, camera_y: int,
                 grayscale: bool=True, margin: int=10, min_confidence: float=0.5, debug_mode: bool=False):
        self.source = source
        self.camera_fov = camera_fov
        self.known_calibration_distance = known_calibration_distance
        self.ball_d_inches = ball_d_inches
        self.known_calibration_pixel_height = known_calibration_pixel_height
        
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

        self.focal_length_pixels = (self.known_calibration_pixel_height * self.known_calibration_distance) / self.ball_d_inches

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
            fps_text = f'FPS: {int(fps)}'
            cv2.putText(annotated_frame, fps_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)

            if self.gui_available: # Fixes a really anooying error I had
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

            if (conf < self.min_confidence):
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

            pt = self.apply_camera_transformations(cx, cy, img_w, img_h, distance_los)
            if pt is None:
                continue

            map_points.append([pt[0], pt[1]])

        return np.array(map_points) if map_points else np.empty((0, 2))

    def apply_camera_transformations(self, cx_pixel, cy_pixel, img_w, img_h, distance_los):
        cx0 = img_w / 2.0
        cy0 = img_h / 2.0
        f = self.focal_length_pixels

        dx = cx_pixel - cx0
        dy = cy_pixel - cy0

        alpha = math.atan2(dx, f)
        beta = math.atan2(dy, f)

        # The pitch angle shifts what we consider "center" vertically
        pitch_rad = math.radians(self.camera_pitch_angle)
        
        # The vertical angle relative to the camera's optical axis (not relative to horizontal)
        # This is just the angle from the image center to the object
        vertical_angle_from_center = beta
        
        # Decompose line-of-sight distance into components
        # The line-of-sight distance is calculated from bounding box size
        # We decompose it based on the angles from image center
        local_x = distance_los * math.cos(vertical_angle_from_center) * math.cos(alpha)
        local_y = distance_los * math.cos(vertical_angle_from_center) * math.sin(alpha)

        yaw_rad = math.radians(self.camera_bot_relative_yaw)
        rot_x = (local_x * math.cos(yaw_rad)) - (local_y * math.sin(yaw_rad))
        rot_y = (local_x * math.sin(yaw_rad)) + (local_y * math.cos(yaw_rad))

        robot_rel_x = rot_x + self.camera_x
        robot_rel_y = rot_y + self.camera_y

        return robot_rel_x, robot_rel_y
        
    def to_field_relative(self, rel_x, rel_y, robot_x, robot_y, robot_heading):
        # this wont work or smth im tweaking just dont use it but keep it here
        # robot_heading should be in degrees i think
        # 0 rad = facing down the field positve y
        # Positive = rotating towards the right.

        f_x = (rel_x * math.cos(robot_heading)) - (rel_y * math.sin(robot_heading))
        f_y = (rel_x * math.sin(robot_heading)) + (rel_y * math.cos(robot_heading))

        field_x = f_x + robot_x
        field_y = f_y + robot_y

        return field_x, field_y
    
    def destroy(self):
        self.cap.release()
        cv2.destroyAllWindows()