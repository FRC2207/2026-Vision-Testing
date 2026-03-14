import cv2
import numpy as np
import time
from rknnlite.api import RKNNLite

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# YOLOv8 Nano anchors and strides (example, adjust to your model)
ANCHORS = np.array([[10,13, 16,30, 33,23],
                    [30,61, 62,45, 59,119],
                    [116,90, 156,198, 373,326]]).reshape(3,3,2)
STRIDES = np.array([8,16,32])  # corresponding to the 3 detection layers

def decode_yolo(out, img_shape, conf_threshold=0.3):
    """Decode YOLOv8 NoNMS raw output"""
    boxes = []
    orig_h, orig_w = img_shape
    for i, layer in enumerate(out):
        # layer shape: [batch, anchors*(5+num_classes), grid_h, grid_w]
        b, c, h, w = layer.shape
        num_anchors = ANCHORS[i].shape[0]
        num_classes = c // num_anchors - 5
        layer = layer.reshape(b, num_anchors, 5+num_classes, h, w)
        layer = layer.transpose(0,1,3,4,2)  # batch, anchors, grid_h, grid_w, 5+classes

        for anchor_idx in range(num_anchors):
            for gy in range(h):
                for gx in range(w):
                    pred = layer[0, anchor_idx, gy, gx]  # 5+classes
                    x, y, w_box, h_box, conf = pred[:5]
                    conf = sigmoid(conf)
                    if conf < conf_threshold:
                        continue
                    x = (sigmoid(x) + gx) * STRIDES[i]
                    y = (sigmoid(y) + gy) * STRIDES[i]
                    w_box = np.exp(w_box) * ANCHORS[i][anchor_idx,0]
                    h_box = np.exp(h_box) * ANCHORS[i][anchor_idx,1]
                    x1, y1 = int(x - w_box/2), int(y - h_box/2)
                    x2, y2 = int(x + w_box/2), int(y + h_box/2)
                    boxes.append([x1, y1, x2, y2, conf])
    return boxes

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

# Feed raw image; RKNN handles preprocessing
img = cv2.imread("Images/1.png")
img_input = np.expand_dims(img, 0)  # batch dimension

print("Input tensor info:", img_input.shape, img_input.dtype, "range:", img_input.min(), img_input.max())

print("\nRunning inference...")
start = time.time()
outputs = rknn.inference(inputs=[img_input])
end = time.time()
print("Inference time: {:.2f} ms".format((end - start) * 1000))

# If multiple detection layers
if isinstance(outputs, list):
    boxes = decode_yolo(outputs, img.shape[:2])
else:
    boxes = decode_yolo([outputs], img.shape[:2])

print("Valid boxes:", len(boxes))
print("First 10 boxes:", boxes[:10])

rknn.release()