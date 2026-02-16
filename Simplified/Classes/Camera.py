import cv2
import math
import numpy as np
from ultralytics import YOLO
import time

try:
    from rknn.api import RKNN
except ImportError:
    RKNN = None

class YoloWrapper:
    def __init__(self, model_file: str, input_size=(640, 640)):
        self.model_file = model_file
        self.input_size = input_size
        self.model_type = None

        if model_file.endswith(".rknn"):
            if RKNN is None:
                raise ImportError("Could node import RKNNLite. This could be because you meant to run a .pt or .onnx on a laptop, but if its the pi ur cooked.")
            self.model_type = "rknn"
            self.model = RKNN()
            ret = self.model.load_rknn(self.model_file)
            if ret != 0:
                raise ValueError(f"Failed to load RKNN model: {self.model_file}")
            
            # Already built if .rknn file so skip
            # ret = self.model.build(do_quantization=False)
            
            # if ret != 0:
            #     raise ValueError(f"Failed to build RKNN model: {self.model_file}")
            
            ret = self.model.init_runtime(target="rk3588")
            if ret != 0:
                raise ValueError(f"Failed to initialize RKNN runtime for model: {self.model_file}")
        elif model_file.endswith(".onnx") or model_file.endswith(".pt"):
            self.model_type = "yolo"
            self.model = YOLO(self.model_file, verbose=False, task="detect")
        else:
            raise ValueError(f"Unsupported model file type: {self.model_file}")
        
    def predict(self, frame):
        if self.model_type == "rknn":
            img = cv2.resize(frame, self.input_size)
            img = img.astype(np.float32) / 255.0
            img = np.expand_dims(img, axis=0)
            outputs = self.model.inference(inputs=[img])[0]

            return self._convert_rknn_outputs(outputs, frame.shape)
        elif self.model_type == "yolo":
            results = self.model(frame, verbose=False)[0]
            return results
        else:
            raise ValueError(f"Unsupported model type: {self.model_type}. This should never happen.")
        
    def _convert_rknn_outputs(self, outputs, orig_shape):
        # I think the _ means like private method but im guessing at this point
        # Hopefully this will make it so that I don't have to chane the rest of the code to work with wtv rknn files output
        class Boxes:
            def __init__(self, xyxy, conf):
                self.xyxy = xyxy
                self.conf = conf

        class Results:
            class BoxesInner:
                def __init__(self, xyxy, conf, frame):
                    self.xyxy = xyxy
                    self.conf = conf
                    self.frame = frame

            def __init__(self, xyxy, conf, orig_shape):
                self.boxes = Boxes(xyxy, conf)
                self.orig_shape = orig_shape

            def plot(self, frame):
                # Custom plotting thingie since rknn doesnt have built in plottin like ultralytics
                for box in self.boxes.xyxy:
                    x1, y1, x2, y2 = map(int, box)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                return frame
            
        outputs = np.array(outputs)
        if len(outputs.shape) == 3: # Prolly (1, N, 6) where N is number of detections and 6 is x1, y1, x2, y2, conf, class. I hate array dimension stuff so much
            outputs = outputs[0]

        xyxy_list = [list(map(float, box[:4])) for box in outputs]
        conf_list = [float(box[4]) for box in outputs]

        return Results(np.array(xyxy_list), np.array(conf_list), orig_shape)

    def release(self):
        if self.model_type == "rknn":
            self.model.release()

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
        self.last_time = time.perf_counter()

        self.cap = cv2.VideoCapture(self.source)
        # self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        # time.sleep(0.5)

        if not self.cap.isOpened():
            raise ValueError(f"Cannot open source {self.source}")

        self.focal_length_pixels = (self.known_calibration_pixel_height * self.known_calibration_distance) / self.ball_d_inches

        self.model = YoloWrapper(self.yolo_model_file)

    def get_frame(self):
        ret, frame = self.cap.read()
        if not ret:
            raise ValueError(f"Failed to retrieve frame from : {self.source}")
        if self.grayscale:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        return frame
                    
    def get_yolo_data(self):
        frame = self.get_frame()
        # If grayscale, convert back to 3 channels for YOLO model
        if self.grayscale:
            frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)

        results = self.model.predict(frame)

        # Show it with cv2
        if self.debug_mode:
            annotated_frame = results.plot(frame)
            new_time_frame = time.perf_counter()
            fps = 1 / (new_time_frame - self.last_time)
            self.last_time = new_time_frame

            # Format and display FPS on the frame
            fps_text = f'FPS: {int(fps)}'
            cv2.putText(annotated_frame, fps_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)

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