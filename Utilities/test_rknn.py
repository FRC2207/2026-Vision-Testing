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

sdk_ver = rknn.get_sdk_version()
print("SDK version:", sdk_ver)

img_bgr = cv2.imread("Images/1.png")
orig_h, orig_w = img_bgr.shape[:2]
img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
img_lb, scale, pad_x, pad_y = letterbox(img_rgb, INPUT_SIZE)
img_input = np.expand_dims(img_lb, axis=0).astype(np.float16)

print("Input tensor:", img_input.shape, img_input.dtype,
      "range:", img_input.min(), "–", img_input.max())

print("\nRunning inference...")
start   = time.time()
outputs = rknn.inference(inputs=[img_input])
end     = time.time()
print(f"Inference time: {(end - start)*1000:.1f} ms")

raw = outputs[0]
print("\nRaw output shape:", raw.shape, "dtype:", raw.dtype)
print("Raw min/max/mean:", raw.min(), raw.max(), raw.mean())

if raw.dtype == np.int8:
    print("Output is int8 — dequantizing with scale=1/128, zero_point=0")
    frame_out = raw[0].astype(np.float32) / 128.0
elif raw.dtype == np.float16:
    frame_out = raw[0].astype(np.float32)
else:
    frame_out = raw[0]

print("After dequant min/max/mean:", frame_out.min(), frame_out.max(), frame_out.mean())

if frame_out.shape[0] == 5 and frame_out.shape[1] > 5:
    frame_out = frame_out.T # [8400, 5]

boxes  = []
scores = []
for row in frame_out:
    x_c, y_c, w, h, conf = row[:5]
    conf = float(conf)
    if conf < CONF_THRESH:
        continue

    x = (x_c - pad_x) / scale
    y = (y_c - pad_y) / scale
    w /= scale
    h /= scale

    x1 = int(x - w / 2)
    y1 = int(y - h / 2)
    x2 = int(x + w / 2)
    y2 = int(y + h / 2)

    if (x2 - x1) <= 0 or (y2 - y1) <= 0:
        continue

    boxes.append([x1, y1, x2 - x1, y2 - y1])
    scores.append(conf)

print(f"\nBoxes before NMS: {len(boxes)}")

if boxes:
    indices = cv2.dnn.NMSBoxes(boxes, scores, score_threshold=CONF_THRESH, nms_threshold=0.45)
    indices = indices.flatten() if len(indices) > 0 else []
    print(f"Boxes after NMS:  {len(indices)}")
    for i in indices:
        x, y, w, h = boxes[i]
        print(f"  [{x},{y},{x+w},{y+h}]  conf={scores[i]:.3f}")
else:
    print("No detections above threshold.")

rknn.release()