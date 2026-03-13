import cv2
import numpy as np
from rknnlite2.api import RKNNLite

# ------------------------------
# CONFIG
# ------------------------------
RKNN_MODEL = "model_test_not_quant.rknn"
IMAGE_PATH = "Images/1.png"
OUTPUT_PATH = "Images/1_out.png"
CONFIDENCE_THRESHOLD = 0.3
IMAGE_WIDTH, IMAGE_HEIGHT = 640, 640  # match training size

# ------------------------------
# LOAD RKNN MODEL
# ------------------------------
rknn = RKNNLite()
rknn.load_rknn(RKNN_MODEL)
rknn.init_runtime(core_mask=RKNNLite.NPU_CORE_0)

# ------------------------------
# LOAD AND RESIZE IMAGE
# ------------------------------
img = cv2.imread(IMAGE_PATH)
img_resized = cv2.resize(img, (IMAGE_WIDTH, IMAGE_HEIGHT))

# ------------------------------
# INFERENCE
# ------------------------------
outputs = rknn.inference(inputs=[img_resized])
raw = outputs[0]  # assuming single output
print("Raw predictions shape:", raw.shape)

# Transpose if needed (check your model)
data = raw.T  # shape: (N, 5)

# ------------------------------
# FILTER INVALID VALUES
# ------------------------------
valid_mask = ~np.isinf(data).any(axis=1) & ~np.isnan(data).any(axis=1)
data = data[valid_mask]

print("Filtered predictions shape:", data.shape)
print("NaN count:", np.isnan(data).sum())
print("Inf count:", np.isinf(data).sum())
print("Sample rows:\n", data[:10])

# ------------------------------
# PROCESS DETECTIONS
# ------------------------------
for i, (x, y, w, h, score) in enumerate(data):
    if score < CONFIDENCE_THRESHOLD:
        continue

    # convert center/size to top-left / bottom-right
    x1 = int(x - w / 2)
    y1 = int(y - h / 2)
    x2 = int(x + w / 2)
    y2 = int(y + h / 2)

    # clip coordinates to image bounds
    x1 = max(0, min(x1, IMAGE_WIDTH - 1))
    y1 = max(0, min(y1, IMAGE_HEIGHT - 1))
    x2 = max(0, min(x2, IMAGE_WIDTH - 1))
    y2 = max(0, min(y2, IMAGE_HEIGHT - 1))

    # draw bounding box
    cv2.rectangle(img_resized, (x1, y1), (x2, y2), (0, 255, 0), 2)
    cv2.putText(img_resized, f"{score:.2f}", (x1, y1-5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

# ------------------------------
# SAVE RESULT (headless)
# ------------------------------
cv2.imwrite(OUTPUT_PATH, img_resized)
print(f"Detection results saved to {OUTPUT_PATH}")

rknn.release()