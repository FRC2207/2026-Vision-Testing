import cv2
import numpy as np
from rknnlite.api import RKNNLite

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# -----------------
# Load RKNN model
# -----------------
rknn = RKNNLite()
rknn.load_rknn("model_test_not_quant.rknn")
rknn.init_runtime(core_mask=RKNNLite.NPU_CORE_0)

# -----------------
# Load and preprocess image
# -----------------
img = cv2.imread("Images/1.png")
orig = img.copy()
img_resized = cv2.resize(img, (640, 640))
img_input = np.expand_dims(img_resized, 0).astype(np.float32)  # Ensure float32
img_input = np.ascontiguousarray(img_input)

# -----------------
# Run inference
# -----------------
outputs = rknn.inference(inputs=[img_input])
raw = outputs[0][0]          # (5, 8400) for example
data = raw.T                 # (8400, 5)

# Filter invalid rows
valid_mask = ~np.isinf(data).any(axis=1) & ~np.isnan(data).any(axis=1)
data = data[valid_mask]

print("Shape after filtering:", data.shape)
print("NaN count:", np.isnan(data).sum())
print("Inf count:", np.isinf(data).sum())
print("Sample rows:\n", data[:10])

# -----------------
# Decode boxes
# -----------------
boxes = []
scores = []

for row in data:
    x, y, w, h, conf = row
    conf = sigmoid(conf)
    if conf < 0.4:   # threshold
        continue
    x1 = int(x - w/2)
    y1 = int(y - h/2)
    x2 = int(x + w/2)
    y2 = int(y + h/2)
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

# Flatten indices if needed
if len(indices) > 0:
    indices = indices.flatten()
else:
    indices = []

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