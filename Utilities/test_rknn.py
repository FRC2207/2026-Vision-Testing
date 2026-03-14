import cv2
import numpy as np
import time
from rknnlite.api import RKNNLite


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def letterbox(img, target_size=(640, 640)):
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

    padded = cv2.copyMakeBorder(
        resized,
        top,
        bottom,
        left,
        right,
        cv2.BORDER_CONSTANT,
        value=(114,114,114)
    )

    return padded, scale, left, top


print("Loading RKNN model...")

rknn = RKNNLite()

ret = rknn.load_rknn("YoloModels/v26/nano/NoNMS-End2End-Half/model_quant.rknn")
if ret != 0:
    print("Failed to load RKNN model")
    exit()

ret = rknn.init_runtime(core_mask=RKNNLite.NPU_CORE_0)
if ret != 0:
    print("Failed to init runtime")
    exit()

print("Runtime initialized")


img = cv2.imread("Images/1.png")

orig_h, orig_w = img.shape[:2]

img_resized, scale, pad_x, pad_y = letterbox(img)

img_input = np.expand_dims(img_resized, 0).astype(np.uint8)
img_input = np.ascontiguousarray(img_input)

print("Input tensor info")
print("shape:", img_input.shape)
print("dtype:", img_input.dtype)
print("range:", img_input.min(), img_input.max())


print("\nRunning inference...")

start = time.time()

outputs = rknn.inference(inputs=[img_input])

end = time.time()

print("Inference time:", (end - start) * 1000, "ms")


out = outputs[0]

print("\nOutput tensor info")
print("shape:", out.shape)
print("dtype:", out.dtype)

print("min/max:", out.min(), out.max())
print("mean:", out.mean())

flat = out.flatten()

print("\nOutput distribution")
print("first 20:", flat[:20])
print("percent zero:", np.mean(flat == 0) * 100, "%")
print("percent negative:", np.mean(flat < 0) * 100, "%")
print("percent positive:", np.mean(flat > 0) * 100, "%")

raw = out[0]

data = raw.T

print("\nDecoded tensor shape:", data.shape)

conf = data[:,4]

print("\nConfidence stats BEFORE sigmoid")
print("min:", conf.min())
print("max:", conf.max())
print("mean:", conf.mean())

sig_conf = sigmoid(conf)

print("\nConfidence stats AFTER sigmoid")
print("min:", sig_conf.min())
print("max:", sig_conf.max())
print("mean:", sig_conf.mean())

print("\nFirst 10 detections:")

for i in range(10):
    print(data[i])

print("\nTesting box decoding...")

boxes = []

for row in data[:200]:

    x, y, w, h, conf = row

    conf = float(conf)

    if conf < 0.3:
        continue

    x = (x - pad_x) / scale
    y = (y - pad_y) / scale

    w /= scale
    h /= scale

    x1 = int(x - w/2)
    y1 = int(y - h/2)
    x2 = int(x + w/2)
    y2 = int(y + h/2)

    if (x2-x1) <= 0 or (y2-y1) <= 0:
        continue

    boxes.append([x1,y1,x2,y2,conf])


print("Valid boxes:", len(boxes))


print("\nFinished testing.")

rknn.release()