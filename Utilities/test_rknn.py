import cv2
import numpy as np
import time
from rknnlite.api import RKNNLite

CONF_THRESH = 0.3
INPUT_SIZE  = (640, 640)

def letterbox(img, target_size=(640, 640)):
    h, w = img.shape[:2]
    tw, th = target_size
    scale = min(tw / w, th / h)
    nw, nh = int(w * scale), int(h * scale)
    resized = cv2.resize(img, (nw, nh))
    pad_w, pad_h = tw - nw, th - nh
    top, left = pad_h // 2, pad_w // 2
    bottom, right = pad_h - top, pad_w - left
    padded = cv2.copyMakeBorder(
        resized, top, bottom, left, right,
        cv2.BORDER_CONSTANT, value=(114, 114, 114)
    )
    return padded, scale, left, top

print("Loading RKNN model...")
rknn = RKNNLite()
ret = rknn.load_rknn("YoloModels/v26/nano/test/model.rknn")
if ret != 0:
    print("Failed to load RKNN model")
    exit()

ret = rknn.init_runtime(core_mask=RKNNLite.NPU_CORE_0_1_2)
if ret != 0:
    print("Failed to init runtime")
    exit()
print("Runtime initialized")

img_bgr = cv2.imread("Images/1.png")
orig_h, orig_w = img_bgr.shape[:2]
img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
img_lb, scale, pad_x, pad_y = letterbox(img_rgb, INPUT_SIZE)
img_input = np.expand_dims(img_lb, axis=0).astype(np.float16)

print("Input tensor:", img_input.shape, img_input.dtype)
print("\nRunning inference...")
start = time.time()
outputs = rknn.inference(inputs=[img_input])
end = time.time()
print(f"Inference time: {(end - start)*1000:.1f} ms")

raw = outputs[0]  # (1, 300, 6)
print("\nOutput shape:", raw.shape, "dtype:", raw.dtype)

detections = raw[0] # (300, 6)

boxes  = []
scores = []
for det in detections:
    x1, y1, x2, y2, conf, cls_id = det

    if conf < CONF_THRESH:
        continue

    x1 = (x1 - pad_x) / scale
    y1 = (y1 - pad_y) / scale
    x2 = (x2 - pad_x) / scale
    y2 = (y2 - pad_y) / scale

    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

    x1 = max(0, x1)
    y1 = max(0, y1)
    x2 = min(orig_w, x2)
    y2 = min(orig_h, y2)

    if (x2 - x1) <= 0 or (y2 - y1) <= 0:
        continue

    boxes.append([x1, y1, x2, y2])
    scores.append(float(conf))

print(f"\nDetections above threshold ({CONF_THRESH}): {len(boxes)}")
for i, (box, score) in enumerate(zip(boxes, scores)):
    print(f"  Box {i}: {box}  conf={score:.3f}")

if boxes:
    annotated = img_bgr.copy()
    for box, score in zip(boxes, scores):
        x1, y1, x2, y2 = box
        cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(annotated, f"{score:.2f}", (x1, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    cv2.imwrite("test_output.png", annotated)
    print("\nAnnotated image saved to test_output.png")

rknn.release()