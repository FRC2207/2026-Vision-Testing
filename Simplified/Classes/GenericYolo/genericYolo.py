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
    def __init__(self, model_file: str, core_mask, input_size=(640, 640), quantized: bool=False, ):
        self.model_file = model_file
        self.input_size = input_size
        self.core_mask = core_mask
        self.model_type = None
        self.logger = logging.getLogger(__name__)

        self._output_fmt   = None  # Auto-detected: "end2end" or "no_nms"
        self._needs_sigmoid = None  # Auto-detected: True if conf column needs sigmoid, False if already 0-1


        self.quantized = quantized
        self.logger.info(
            f"YoloWrapper init: model={model_file}, input_size={input_size}, quantized={quantized}"
        )
        
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
            ret = self.model.init_runtime(core_mask=core_mask)
            # core_mask=RKNNLite.NPU_CORE_0_1_2 use for all NPU usage
            if ret != 0:
                self.logger.error(
                    f"Failed to initialize RKNN runtime for model: {self.model_file}"
                )
                raise ValueError(
                    f"Failed to initialize RKNN runtime for model: {self.model_file}"
                )

            # Pre-allocate the input buffer once so we dont malloc every frame
            # input_size is (W, H) so flip it for the numpy shape
            h, w = self.input_size[1], self.input_size[0]
            self._input_buf = np.empty((1, h, w, 3), dtype=np.uint8)

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

    def _preprocess_for_rknn(self, frame: np.ndarray) -> np.ndarray:
        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # Write directly into the pre-allocated buffer instead of allocating a new one every frame
        self._letterbox_into(img_rgb, self._input_buf[0], self.input_size)
        return self._input_buf

    def _letterbox(self, img: np.ndarray, target_size: tuple) -> tuple:
        h, w = img.shape[:2]
        target_w, target_h = target_size
        scale = min(target_w / w, target_h / h)
        new_w, new_h = int(w * scale), int(h * scale)
        resized = cv2.resize(img, (new_w, new_h))

        pad_w  = target_w - new_w
        pad_h  = target_h - new_h
        top    = pad_h // 2
        bottom = pad_h - top
        left   = pad_w // 2
        right  = pad_w - left
        padded = cv2.copyMakeBorder(
            resized, top, bottom, left, right,
            cv2.BORDER_CONSTANT, value=(114, 114, 114)
        )
        return padded, scale, left, top

    def _letterbox_into(self, img: np.ndarray, dst: np.ndarray, target_size: tuple) -> None:
        # Same as _letterbox but writes into dst in-place instead of allocating a new array
        # dst should be a (H, W, 3) uint8 buffer
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
        dst[:] = 114  # fill padding color
        dst[top:top + new_h, left:left + new_w] = resized

    def _run_rknn(self, preprocessed: np.ndarray, orig_shape) -> "Results":
        # Extracted so both predict() and predict_preprocessed() can share this without copy pasting
        raw_outputs = self.model.inference(inputs=[preprocessed])

        if raw_outputs is None:
            self.logger.error(
                f"RKNN inference returned None (frame shape={orig_shape}) — skipping."
            )
            return Results([], orig_shape)

        output_tensor = raw_outputs[0] # keep batch dim for format detection

        # Auto-detect output format on first successful inference
        if self._output_fmt is None:
            _, d1, d2 = output_tensor.shape
            self._output_fmt = "end2end" if d2 == 6 else "no_nms"
            self.logger.info(
                f"Detected RKNN output format: {self._output_fmt}, "
                f"shape={output_tensor.shape}, dtype={output_tensor.dtype}"
            )

        # Dequantize if needed
        if output_tensor.dtype == np.int8:
            output_tensor = output_tensor.astype(np.float32) / 128.0
        elif output_tensor.dtype == np.uint8:
            output_tensor = output_tensor.astype(np.float32) / 255.0

        if self._output_fmt == "end2end":
            return self._convert_rknn_end2end_outputs(output_tensor[0], orig_shape)
        else:
            return self._convert_rknn_outputs(output_tensor[0], orig_shape)

    def predict_preprocessed(self, preprocessed: np.ndarray, orig_shape) -> "Results":
        # Fast path used by Camera's pipeline thread — skips preproc since Camera
        # already did it concurrently with the last inference
        if self.model_type != "rknn":
            raise RuntimeError("predict_preprocessed is only valid for RKNN models.")
        return self._run_rknn(preprocessed, orig_shape)

    def predict(self, frame_or_frames, orig_shape=None) -> "Results | list[Results]":
        is_list = isinstance(frame_or_frames, list)
        frames = frame_or_frames if is_list else [frame_or_frames]

        results_list = []

        if self.model_type == "rknn":
            for frame in frames:
                target_shape = orig_shape if orig_shape is not None else frame.shape
                preprocessed = self._preprocess_for_rknn(frame)
                results_list.append(self._run_rknn(preprocessed, target_shape))

        else:  # For these ultralytics handles its own preprocessing
            for frame in frames:
                # imgsz accepts (h, w), ultralytics will do its own letterboxing internally
                result = self.model(
                    frame,
                    verbose=False,
                    imgsz=(self.input_size[1], self.input_size[0]),
                )
                results_list.append(self._convert_ultralytics_to_results(result[0]))

        return results_list if is_list else results_list[0]

    def _convert_rknn_outputs(self, frame_output: np.ndarray, orig_shape) -> Results:
        if frame_output.ndim == 3:
            frame_output = frame_output[0]

        # Ensure shape is [num_boxes, 5]
        if frame_output.shape[0] == 5 and frame_output.shape[1] > 5:
            frame_output = frame_output.T

        # Sigmoid on objectness column
        if self._needs_sigmoid is None:
            sample = frame_output[:, 4]
            self._needs_sigmoid = bool(sample.min() < -0.1 or sample.max() > 1.1)
            self.logger.info(
                f"Confidence activation: {'sigmoid (raw logits detected)' if self._needs_sigmoid else 'direct (pre-activated 0-1 detected)'}"
            )

        if self._needs_sigmoid:
            confs = 1 / (1 + np.exp(-np.clip(frame_output[:, 4], -88, 88)))
        else:
            confs = frame_output[:, 4].copy()

        conf_mask = confs >= 0.5
        frame_output = frame_output[conf_mask]
        confs        = confs[conf_mask]

        if len(frame_output) == 0:
            self.logger.info("No boxes passed confidence threshold.")
            return Results([], orig_shape)
        
        valid_mask = (
            ~np.isinf(frame_output).any(axis=1)
            & ~np.isnan(frame_output).any(axis=1)
        )
        frame_output = frame_output[valid_mask]
        confs        = confs[valid_mask]

        if len(frame_output) == 0:
            return Results([], orig_shape)

        # Remap from letterboxed space back to original image space
        orig_h, orig_w = orig_shape[:2]
        target_w, target_h = self.input_size
        scale = min(target_w / orig_w, target_h / orig_h)
        new_w  = int(orig_w * scale)
        new_h  = int(orig_h * scale)
        pad_x  = (target_w - new_w) / 2
        pad_y  = (target_h - new_h) / 2

        x_c = (frame_output[:, 0] - pad_x) / scale
        y_c = (frame_output[:, 1] - pad_y) / scale
        w   = frame_output[:, 2] / scale
        h   = frame_output[:, 3] / scale

        x1s = np.clip((x_c - w / 2).astype(int), 0, orig_w)
        y1s = np.clip((y_c - h / 2).astype(int), 0, orig_h)
        x2s = np.clip((x_c + w / 2).astype(int), 0, orig_w)
        y2s = np.clip((y_c + h / 2).astype(int), 0, orig_h)

        size_mask = (x2s - x1s > 0) & (y2s - y1s > 0)
        x1s, y1s, x2s, y2s, confs = (
            x1s[size_mask], y1s[size_mask],
            x2s[size_mask], y2s[size_mask],
            confs[size_mask],
        )

        if len(x1s) == 0:
            return Results([], orig_shape)

        boxes  = [
            Box([x1, y1, x2, y2], float(c))
            for x1, y1, x2, y2, c in zip(x1s, y1s, x2s, y2s, confs)
        ]
        scores = confs.tolist()

        nms_boxes = [
            [b.xyxy[0], b.xyxy[1], b.xyxy[2] - b.xyxy[0], b.xyxy[3] - b.xyxy[1]]
            for b in boxes
        ]
        indices = cv2.dnn.NMSBoxes(nms_boxes, scores, score_threshold=0.5, nms_threshold=0.3)
        indices = indices.flatten() if len(indices) > 0 else []

        return Results([boxes[i] for i in indices], orig_shape)

    def _convert_ultralytics_to_results(self, ultralytics_result):
        boxes = [Box(b.xyxy[0].tolist(), float(b.conf)) for b in ultralytics_result.boxes]
        return Results(boxes, ultralytics_result.orig_shape)

    def _sigmoid(self, x):
        return 1 / (1 + np.exp(-x))
    
    def _convert_rknn_end2end_outputs(self, detections: np.ndarray, orig_shape) -> Results:
        orig_h, orig_w = orig_shape[:2]
        target_w, target_h = self.input_size
        scale = min(target_w / orig_w, target_h / orig_h)
        new_w  = int(orig_w * scale)
        new_h  = int(orig_h * scale)
        pad_x  = (target_w - new_w) / 2
        pad_y  = (target_h - new_h) / 2

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