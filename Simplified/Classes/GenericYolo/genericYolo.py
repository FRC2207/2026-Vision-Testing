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
        # self.logger.info(f"Running with is_list: {is_list}")

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
        self.logger.info(f"Original frame_output shape: {frame_output.shape}")

        # Squeeze batch dimension if present
        if frame_output.ndim == 3:
            frame_output = frame_output[0]
            self.logger.info(f"Squeezed frame_output shape: {frame_output.shape}")

        # Transpose RKNN output to [num_boxes, 5] if needed
        if frame_output.shape[0] == 5 and frame_output.shape[1] > 5:
            frame_output = frame_output.T
            self.logger.info(f"Transposed frame_output shape: {frame_output.shape}")

        # Remove invalid rows
        valid_mask = ~np.isinf(frame_output).any(axis=1) & ~np.isnan(frame_output).any(axis=1)
        frame_output = frame_output[valid_mask]
        self.logger.info(f"After filtering invalid rows, frame_output shape: {frame_output.shape}")

        # Compute letterbox padding and scaling
        orig_h, orig_w = orig_shape[:2]
        target_w, target_h = self.input_size
        scale = min(target_w / orig_w, target_h / orig_h)
        new_w, new_h = int(orig_w * scale), int(orig_h * scale)
        pad_x = (target_w - new_w) / 2
        pad_y = (target_h - new_h) / 2

        boxes = []
        scores = []

        # Process each box
        for i, row in enumerate(frame_output):
            x_c, y_c, w, h, conf = row[:5]
            if conf == 0:
                continue
            
            # RKNN model already outputs confidence in [0,1]
            conf = float(conf)

            # Debug: log first few boxes
            if i < 10:
                self.logger.info(f"Box {i}: raw={row}, conf={conf:.4f}")

            if conf < 0.1:  # working threshold
                continue

            # Map coordinates back to original image
            x = (x_c - pad_x) / scale
            y = (y_c - pad_y) / scale
            w /= scale
            h /= scale

            x1 = max(0, int(x - w / 2))
            y1 = max(0, int(y - h / 2))
            x2 = min(orig_w - 1, int(x + w / 2))
            y2 = min(orig_h - 1, int(y + h / 2))

            if (x2 - x1) <= 0 or (y2 - y1) <= 0:
                continue

            boxes.append(Box([x1, y1, x2, y2], conf))
            scores.append(conf)

        self.logger.info(f"Number of boxes before NMS: {len(boxes)}")

        if not boxes:
            self.logger.info("No boxes passed confidence threshold.")
            return Results([], orig_shape)

        # Apply NMS using x,y,w,h format
        nms_boxes = [[b.xyxy[0], b.xyxy[1], b.xyxy[2]-b.xyxy[0], b.xyxy[3]-b.xyxy[1]] for b in boxes]
        indices = cv2.dnn.NMSBoxes(nms_boxes, scores, score_threshold=0.1, nms_threshold=0.45)
        indices = indices.flatten() if len(indices) > 0 else []

        self.logger.info(f"Indices after NMS: {indices}")
        final_boxes = [boxes[i] for i in indices]
        self.logger.info(f"Number of final boxes: {len(final_boxes)}")

        return Results(final_boxes, orig_shape)

    def _convert_ultralytics_to_results(self, ultralytics_result):
        boxes = [Box(list(b.xyxy[0]), float(b.conf[0])) for b in ultralytics_result.boxes]
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

    def release(self):
        if self.model_type == "rknn":
            self.model.release()