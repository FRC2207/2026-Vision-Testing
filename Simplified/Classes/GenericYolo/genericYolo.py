import logging
import cv2
import numpy as np
from ultralytics import YOLO

try:
    from rknn.api import RKNN
    RKNN_FOUND = True
except ImportError:
    RKNN_FOUND = None

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
            if RKNN_FOUND is None:
                self.logger.error(
                    "Could node import RKNN. This could be because you meant to run a .pt or .onnx on a laptop, but if its the pi ur cooked."
                )
                raise ImportError(
                    "Could node import RKNN. This could be because you meant to run a .pt or .onnx on a laptop, but if its the pi ur cooked."
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

            ret = self.model.init_runtime(target="rk3588", core_mask=RKNN.NPU_CORE_0_1_2)
            if ret != 0:
                self.logger.error(
                    f"Failed to initialize RKNN runtime for model: {self.model_file}"
                )
                raise ValueError(
                    f"Failed to initialize RKNN runtime for model: {self.model_file}"
                )
        elif model_file.endswith(".onnx") or model_file.endswith(".pt") or model_file.endswith("openvino_model"):
            self.model_type = "yolo"
            self.model = YOLO(self.model_file, verbose=False, task="detect")
        else:
            self.logger.error(
                f"Unsupported model file type: {self.model_file}. Check constants and spelling blud."
            )
            raise ValueError(
                f"Unsupported model file type: {self.model_file}. Check constants and spelling blud."
            )

    def predict(self, frame_or_frames) -> list[Results]:
        is_list = isinstance(frame_or_frames, list)
        frames = frame_or_frames if is_list else [frame_or_frames]
        
        if self.model_type == "rknn":
            processed = [cv2.resize(f, self.input_size) for f in frames]
            batch_input = np.stack(processed, axis=0)
            
            raw_outputs = self.model.inference(inputs=[batch_input], data_format="nhwc", device='/dev/dri/renderD128')[0]
            print(raw_outputs)
            results_list = [
                self._convert_rknn_outputs(raw_outputs[i], frames[i].shape) 
                for i in range(len(frames))
            ]

        else: # All other types (hopefully) work natevily with Ultralytics YOLO
            results = self.model(frames, verbose=False, imgsz=320)
            results_list = [self._convert_ultralytics_to_results(r) for r in results]

        return results_list if is_list else results_list[0]

    def _convert_rknn_outputs(self, frame_output, orig_shape):
        mask = frame_output[:, 4] > 0.5
        valid = frame_output[mask]
        
        boxes = [Box(list(map(float, b[:4])), float(b[4])) for b in valid]
        return Results(boxes, orig_shape)
    
    def release(self):
        if self.model_type == "rknn":
            self.model.release()