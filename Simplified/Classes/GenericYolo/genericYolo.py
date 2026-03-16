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
    
    def __str__(self):
        s = f"Results(orig_shape={self.orig_shape}, num_boxes={len(self.boxes)})\n"
        for i, box in enumerate(self.boxes):
            s += f"  Box {i}: xyxy={box.xyxy}, conf={box.conf:.3f}\n"
        return s
    
class YoloWrapper:
    def __init__(self, model_file: str, input_size=(640, 640)):
        self.model_file = model_file
        self.input_size = input_size
        self.model_type = None
        self.logger = logging.getLogger(__name__)

        self._output_fmt = None # Used for no-nms and nms models (Yolov11 - Yolov26)

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
            ret = self.model.init_runtime(core_mask=RKNNLite.NPU_CORE_0_1_2)
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

    def predict(self, frame_or_frames, orig_shape=None) -> list[Results]:
        is_list = isinstance(frame_or_frames, list)
        frames = frame_or_frames if is_list else [frame_or_frames]
        # self.logger.info(f"Running with is_list: {is_list}")

        results_list = []

        if self.model_type == "rknn":
            for frame in frames:
                # Inference
                # self.logger.info(f"Frame info: shape={frame.shape}, dtype={frame.dtype}")
                raw_outputs = self.model.inference(inputs=[frame])
                # self.logger.info("Ran inference.")
                output_tensor = raw_outputs[0] # keep batch dim for format detection

                # Auto-detect output format on first inference
                if self._output_fmt is None:
                    _, d1, d2 = output_tensor.shape
                    if d2 == 6:
                        self._output_fmt = "end2end" # (1, 300, 6) - YOLO26 default
                    else:
                        self._output_fmt = "no_nms" # (1, 5, 8400) - YOLO11 / stripped
                    self.logger.info(f"Detected RKNN output format: {self._output_fmt}, shape={output_tensor.shape}, dtype={output_tensor.dtype}")

                if output_tensor.dtype == np.int8: # Dequantize if int8
                    output_tensor = output_tensor.astype(np.float32) / 128.0

                target_shape = orig_shape if orig_shape is not None else frame.shape

                if self._output_fmt == "end2end":
                    results_list.append(
                        self._convert_rknn_end2end_outputs(
                            output_tensor[0],
                            target_shape
                        )
                    )
                else:
                    results_list.append(
                        self._convert_rknn_outputs(
                            output_tensor[0],
                            target_shape
                        )
                    )
        else:
            results = self.model(frames, verbose=False, imgsz=self.input_size[0])
            results_list = [self._convert_ultralytics_to_results(r) for r in results]

        return results_list if is_list else results_list[0]

    def _convert_rknn_outputs(self, frame_output, orig_shape):
        # Squeeze batch dimension if present
        if frame_output.ndim == 3:
            frame_output = frame_output[0]

        # Transpose to [num_boxes, 5] if needed
        if frame_output.shape[0] == 5 and frame_output.shape[1] > 5:
            frame_output = frame_output.T

        # Remove invalid rows
        valid_mask = ~np.isinf(frame_output).any(axis=1) & ~np.isnan(frame_output).any(axis=1)
        frame_output = frame_output[valid_mask]

        # Vectorized sigmoid on conf column — no Python loop
        confs = 1 / (1 + np.exp(-frame_output[:, 4]))

        # Filter by confidence threshold before doing anything else
        conf_mask = confs >= 0.5  # raised from 0.1
        frame_output = frame_output[conf_mask]
        confs = confs[conf_mask]

        if len(frame_output) == 0:
            self.logger.info("No boxes passed confidence threshold.")
            return Results([], orig_shape)

        # Vectorized coordinate remapping
        orig_h, orig_w = orig_shape[:2]
        target_w, target_h = self.input_size
        scale = min(target_w / orig_w, target_h / orig_h)
        new_w, new_h = int(orig_w * scale), int(orig_h * scale)
        pad_x = (target_w - new_w) / 2
        pad_y = (target_h - new_h) / 2

        x_c = (frame_output[:, 0] - pad_x) / scale
        y_c = (frame_output[:, 1] - pad_y) / scale
        w   = frame_output[:, 2] / scale
        h   = frame_output[:, 3] / scale

        x1s = (x_c - w / 2).astype(int)
        y1s = (y_c - h / 2).astype(int)
        x2s = (x_c + w / 2).astype(int)
        y2s = (y_c + h / 2).astype(int)

        # Filter zero-size boxes
        size_mask = (x2s - x1s > 0) & (y2s - y1s > 0)
        x1s, y1s, x2s, y2s, confs = x1s[size_mask], y1s[size_mask], x2s[size_mask], y2s[size_mask], confs[size_mask]

        if len(x1s) == 0:
            return Results([], orig_shape)

        boxes = [Box([x1, y1, x2, y2], float(c)) for x1, y1, x2, y2, c in zip(x1s, y1s, x2s, y2s, confs)]
        scores = confs.tolist()

        # Tighter NMS — 0.3 instead of 0.45 to kill the overlapping grid
        nms_boxes = [[b.xyxy[0], b.xyxy[1], b.xyxy[2]-b.xyxy[0], b.xyxy[3]-b.xyxy[1]] for b in boxes]
        indices = cv2.dnn.NMSBoxes(nms_boxes, scores, score_threshold=0.5, nms_threshold=0.3)
        indices = indices.flatten() if len(indices) > 0 else []

        return Results([boxes[i] for i in indices], orig_shape)

    def _convert_ultralytics_to_results(self, ultralytics_result):
        boxes = [Box(b.xyxy[0].tolist(), float(b.conf)) for b in ultralytics_result.boxes]
        return Results(boxes, ultralytics_result.orig_shape)

    def _sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def letterbox(self, img, target_size=(640, 640)):
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
    
    def _convert_rknn_end2end_outputs(self, detections, orig_shape):
        orig_h, orig_w = orig_shape[:2]
        target_w, target_h = self.input_size
        scale = min(target_w / orig_w, target_h / orig_h)
        new_w, new_h = int(orig_w * scale), int(orig_h * scale)
        pad_x = (target_w - new_w) / 2
        pad_y = (target_h - new_h) / 2

        boxes = []
        for det in detections:
            x1, y1, x2, y2, conf = det[0], det[1], det[2], det[3], det[4]
            conf = float(conf)
            if conf < 0.1:
                continue

            x1 = max(0, int((x1 - pad_x) / scale))
            y1 = max(0, int((y1 - pad_y) / scale))
            x2 = min(orig_w, int((x2 - pad_x) / scale))
            y2 = min(orig_h, int((y2 - pad_y) / scale))

            if (x2 - x1) <= 0 or (y2 - y1) <= 0:
                continue

            boxes.append(Box([x1, y1, x2, y2], conf))

        if not boxes:
            self.logger.info("No boxes passed confidence threshold.")
            return Results([], orig_shape)

        return Results(boxes, orig_shape)

    def release(self):
        if self.model_type == "rknn":
            self.model.release()