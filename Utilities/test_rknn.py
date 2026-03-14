import cv2
import numpy as np
import time
from rknnlite.api import RKNNLite

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# Confidence threshold
CONF_THRESH = 0.3

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

# Load image and resize to model's expected input
img = cv2.imread("Images/1.png")
orig_h, orig_w = img.shape[:2]
img_resized = cv2.resize(img, (640, 640))
img_input = np.expand_dims(img_resized, 0)

print("Input tensor info:", img_input.shape, img_input.dtype, "range:", img_input.min(), img_input.max())

print("\nRunning inference...")
start = time.time()
outputs = rknn.inference(inputs=[img_input])
end = time.time()
print("Inference time: {:.2f} ms".format((end - start) * 1000))

out = outputs[0]  # shape: [num_boxes, 5] for NoNMS
print("\nOutput tensor info:", out.shape, out.dtype)
print("min/max:", out.min(), out.max(), "mean:", out.mean())

# Apply sigmoid on confidence
boxes = []
for row in out:
    x, y, w, h, conf, cls_id = row
    conf = sigmoid(conf)
    if conf < CONF_THRESH:
        continue

    # Scale boxes back to original image size
    x = x * orig_w / 640
    y = y * orig_h / 640
    w = w * orig_w / 640
    h = h * orig_h / 640
    x1, y1 = int(x - w/2), int(y - h/2)
    x2, y2 = int(x + w/2), int(y + h/2)

    boxes.append([x1, y1, x2, y2, conf])

print("Valid boxes:", len(boxes))
print("First 10 boxes:", boxes[:10])

rknn.release()