import cv2
import numpy as np
from rknnlite.api import RKNNLite

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def letterbox(img, target_size=(640, 640)):
    """Resize image with unchanged aspect ratio using padding"""
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

# -----------------
# Load RKNN model
# -----------------
rknn = RKNNLite()
rknn.load_rknn("model_test.rknn")
rknn.init_runtime(core_mask=RKNNLite.NPU_CORE_0)

# -----------------
# Load and preprocess image
# -----------------
img = cv2.imread("Images/1.png")
orig_h, orig_w = img.shape[:2]
orig = img.copy()

img_resized, scale, pad_x, pad_y = letterbox(img, (640, 640))
img_input = np.expand_dims(img_resized, 0).astype(np.float32)
img_input = np.ascontiguousarray(img_input)

# -----------------
# Run inference
# -----------------
outputs = rknn.inference(inputs=[img_input])
raw = outputs[0][0]          
data = raw.T                 

# Filter invalid rows
valid_mask = ~np.isinf(data).any(axis=1) & ~np.isnan(data).any(axis=1)
data = data[valid_mask]

print("Shape after filtering:", data.shape)

# -----------------
# Decode boxes
# -----------------
boxes = []
scores = []

for row in data:
    x, y, w, h, conf = row
    if conf == 0:  # Remove zeros before sigmoid
        continue
    conf = sigmoid(conf)
    if conf < 0.6:
        continue

    # Convert from model coordinates to original image coordinates
    x = (x - pad_x) / scale
    y = (y - pad_y) / scale
    w /= scale
    h /= scale

    x1 = int(x - w/2)
    y1 = int(y - h/2)
    x2 = int(x + w/2)
    y2 = int(y + h/2)

    x1 = max(0, min(x1, orig_w - 1))
    y1 = max(0, min(y1, orig_h - 1))
    x2 = max(0, min(x2, orig_w - 1))
    y2 = max(0, min(y2, orig_h - 1))

    if (x2 - x1) <= 0 or (y2 - y1) <= 0:
        continue

    boxes.append([x1, y1, x2-x1, y2-y1])
    scores.append(float(conf))

print("Candidates before NMS:", len(boxes))

# -----------------
# Non-Max Suppression
# -----------------
indices = cv2.dnn.NMSBoxes(
    boxes,
    scores,
    score_threshold=0.4,
    nms_threshold=0.45
)

indices = indices.flatten() if len(indices) > 0 else []
print("Final boxes after NMS:", len(indices))

# -----------------
# Draw results
# -----------------
for i in indices:
    x, y, w, h = boxes[i]
    cv2.rectangle(orig, (x, y), (x+w, y+h), (0, 255, 0), 2)
    cv2.putText(
        orig,
        f"{scores[i]:.2f}",
        (x, y-5),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (0, 255, 0),
        2
    )

cv2.imwrite("detected.png", orig)
rknn.release()