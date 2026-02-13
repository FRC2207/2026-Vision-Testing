import cv2
import math
import numpy as np
from ultralytics import YOLO
import requests

class Camera:
    def __init__(self, source: int|str, camera_fov: int, known_calibration_distance: int, ball_d_inches: int, known_calibration_pixel_height: int, yolo_model_file: str, camera_angle: float, camera_height: int, grayscale: bool=True, margin: int=10, min_confidence: float=0.5, debug_mode: bool=False):
        self.source = source
        self.camera_fov = camera_fov
        self.known_calibration_distance = known_calibration_distance
        self.ball_d_inches = ball_d_inches
        self.known_calibration_pixel_height = known_calibration_pixel_height
        
        self.margin = margin
        self.min_confidence = min_confidence
        self.grayscale = grayscale
        self.yolo_model_file = yolo_model_file

        self.camera_angle = camera_angle
        self.camera_height = camera_height

        self.debug_mode = debug_mode

        self.cap = cv2.VideoCapture(self.source)

        if not self.cap.isOpened():
            raise ValueError(f"Cannot open source {self.source}")

        self.focal_length_pixels = (self.known_calibration_pixel_height * self.known_calibration_distance) / self.ball_d_inches

    def get_frame(self):
        ret, frame = self.cap.read()
        if not ret:
            raise ValueError(f"Failed to retrieve frame from {self.video_feed_type}: {self.source}")
        if self.grayscale:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        return frame
                    
    def get_yolo_data(self):
        frame = self.get_frame()
        # If grayscale, convert back to 3 channels for YOLO model
        if self.grayscale:
            frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
        model = YOLO(self.yolo_model_file, verbose=False)
        results = model(frame, verbose=False)[0]

        # Show it with cv2
        if self.debug_mode:
            annotated_frame = results.plot()
            cv2.imshow("YOLO Detections", annotated_frame)
            cv2.waitKey(1)

        return results

    def run(self):
        data = self.get_yolo_data()
        img_h, img_w = data.orig_shape[:2]

        map_points = []
        confidence_list = []

        for i, box in enumerate(data.boxes.xyxy):
            x1, y1, x2, y2 = box.tolist()

            conf = float(data.boxes.conf[i].item())

            if (conf < self.min_confidence):
                continue
            # Skip boxes too close to left/top/right edges hopefully allowing bottem edge
            if x1 < self.margin or y1 < self.margin or x2 > (img_w - self.margin):
                continue

            confidence_list.append(conf)
            
            cx = (x1 + x2) / 2
            cy = (y1 + y2) / 2
            w_pixels = x2 - x1
            h_pixels = y2 - y1
            avg_pixels = (w_pixels + h_pixels) / 2
            distance = (self.ball_d_inches * self.focal_length_pixels) / avg_pixels
            
            dx_pixels = cx - (img_w / 2)
            angle_rad = math.radians(self.camera_fov) * (dx_pixels / img_w)
            
            X = math.tan(angle_rad) * distance
            Y = distance

            X, Y = self.apply_camera_transformations(X, Y)
            
            map_points.append([X, Y])
        
        return np.array(map_points) if map_points else np.empty((0, 2))
    
    def apply_camera_transformations(self, X, Y):
        # Fancy math, but it came from stack overflow and it calulates the real world X and Y coordinates based on the camera angle and height
        ground_distance = self.camera_height / math.cos(math.radians(self.camera_angle))
        Y += ground_distance * math.cos(math.radians(self.camera_angle))
        return X, Y

    def destroy(self):
        self.cap.release()
        cv2.destroyAllWindows()