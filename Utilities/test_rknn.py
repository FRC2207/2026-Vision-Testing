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
    top, bottom = pad_h // 2, pad_h - (pad_h // 2)
    left, right = pad_w // 2, pad_w - (pad_w // 2)
    padded = cv2.copyMakeBorder(resized, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(114,114,114))
    return padded, scale, left, top

print("Loading RKNN model...")
rknn = RKNNLite()
ret = rknn.load_rknn("YoloModels/v26/nano/NoNMS/model.rknn")
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
img_input = img_resized.astype(np.uint8)
img_input = img_input.transpose(2,0,1)
img_input = np.expand_dims(img_input, 0)

print("Input tensor info:", img_input.shape, img_input.dtype, "range:", img_input.min(), img_input.max())

print("\nRunning inference...")
start = time.time()
outputs = rknn.inference(inputs=[img_input])
end = time.time()
print("Inference time: {:.2f} ms".format((end - start) * 1000))

out = outputs[0]
print("\nOutput tensor info:", out.shape, out.dtype)
print("min/max:", out.min(), out.max(), "mean:", out.mean())

flat = out.flatten()
print("Output distribution - first 20:", flat[:20])
print("percent zero:", np.mean(flat == 0) * 100, "%")
print("percent negative:", np.mean(flat < 0) * 100, "%")
print("percent positive:", np.mean(flat > 0) * 100, "%")

data = out[0].T  # shape: [num_boxes, 5]
conf = data[:,4]
sig_conf = sigmoid(conf)

print("\nConfidence stats BEFORE sigmoid:", conf.min(), conf.max(), conf.mean())
print("Confidence stats AFTER sigmoid:", sig_conf.min(), sig_conf.max(), sig_conf.mean())

print("\nFirst 10 detections:")
for i, row in enumerate(data):
    print(row)
    if i >= 9:  # stop after 10 rows
        break

print("\nDecoding boxes (manual, no NMS)...")
boxes = []
for row in data:
    x, y, w, h, c = row
    c = float(sigmoid(c))  # use sigmoid confidence
    if c < 0.3:  # threshold
        continue

    # scale back to original image
    x = (x - pad_x) / scale
    y = (y - pad_y) / scale
    w /= scale
    h /= scale

    x1, y1 = int(x - w/2), int(y - h/2)
    x2, y2 = int(x + w/2), int(y + h/2)

    if x2 <= x1 or y2 <= y1:
        continue

    boxes.append([x1, y1, x2, y2, c])

print("Valid boxes:", len(boxes))
print("\nFinished testing.")

rknn.release()