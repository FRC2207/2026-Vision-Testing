import logging
import cv2
import numpy as np
from ultralytics import YOLO

try:
    from rknnlite.api import RKNNLite
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
            self.model = RKNNLite()

            # Load the .rknn model file
            ret = self.model.load_rknn(self.model_file)
            if ret != 0:
                self.logger.error(f"Failed to load RKNN model: {self.model_file}")
                raise ValueError(f"Failed to load RKNN model: {self.model_file}")

            # Already built if .rknn file so skip
            # ret = self.model.build(do_quantization=False)
            # if ret != 0:
            #     raise ValueError(f"Failed to build RKNN model: {self.model_file}")

            # Initialize the RKNN runtime on NPU 0
            ret = self.model.init_runtime(core_mask=RKNNLite.NPU_CORE_0)
            # core_mask=RKNNLite.NPU_CORE_0_1_2 use for all NPU usage
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
        self.logger.info(f"Running with is_list: {is_list}")

        results_list = []

        if self.model_type == "rknn":
            for frame in frames:
                # Inference
                # self.logger.info(f"Frame info: shape={frame.shape}, dtype={frame.dtype}")
                raw_outputs = self.model.inference(inputs=[frame])
                # self.logger.info("Ran inference.")
                output_tensor = raw_outputs[0][0]
                # self.logger.info(f"Output: {output_tensor}")
                # self.logger.info(f"Raw output: {raw_outputs}")

                results_list.append(
                    self._convert_rknn_outputs(
                        output_tensor,
                        frame.shape
                    )
                )
        else:
            results = self.model(frames, verbose=False, imgsz=self.input_size[0])
            results_list = [self._convert_ultralytics_to_results(r) for r in results]

        return results_list if is_list else results_list[0]

    def _convert_rknn_outputs(self, frame_output, orig_shape):
        # 1. Assuming index 4 IS your confidence column (even if it currently shows 0.0)
        # Apply Sigmoid to turn raw model output into a probability
        confidences = self._sigmoid(frame_output[:, 4]) 
        
        # 2. Check the raw values BEFORE sigmoid
        self.logger.info(f"Raw Logit (col 4): {frame_output[0, 4]}")
        self.logger.info(f"Sigmoid Confidence: {confidences[0]}")
        
        # 3. Use the probability for your mask
        mask = confidences > 0.5
        valid = frame_output[mask]
        
        # 4. Create Box objects
        boxes = [Box(list(map(float, b[:4])), float(confidences[i])) for i, b in enumerate(frame_output) if mask[i]]
        return Results(boxes, orig_shape)

    def _convert_ultralytics_to_results(self, ultralytics_result):
        boxes = [Box(list(b.xyxy[0]), float(b.conf[0])) for b in ultralytics_result.boxes]
        return Results(boxes, ultralytics_result.orig_shape)

    def _sigmoid(x):
        return 1 / (1 + np.exp(-x))

    def release(self):
        if self.model_type == "rknn":
            self.model.release()