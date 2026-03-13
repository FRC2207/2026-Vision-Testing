from rknnlite.api import RKNNLite
import cv2
import numpy as np

# Initialize
rknn = RKNNLite()
rknn.load_rknn('model_test_backup.rknn')
rknn.init_runtime(core_mask=RKNNLite.NPU_CORE_0)

# Preprocess (CRITICAL: Match your training normalization)
img = cv2.imread('Images/1.png')
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
img = cv2.resize(img, (640, 640))
img = np.expand_dims(img, axis=0)

# Inference
outputs = rknn.inference(inputs=[img])

# DIAGNOSTIC: Look at the data
print(f"Number of output tensors: {len(outputs)}")
print(f"Shape of first output: {outputs[0].shape}")
# If this shape is (1, 300, 85) or similar, the model is working!
# If it is (1, 8400, ...), you are reading the raw grid.
print(f"Sample raw values (first 10): {outputs[0][0][:10]}")

# Assuming outputs[0] is (1, 5, 8400)
raw_data = outputs[0][0] # Shape: (5, 8400)
data = raw_data.T        # Shape: (8400, 5) -> [x, y, w, h, confidence]

# 1. Filter by confidence
CONF_THRESHOLD = 0.5 
# This line keeps only rows where the 5th column (index 4) > 0.5
valid_detections = data[data[:, 4] > CONF_THRESHOLD]

print(f"Found {len(valid_detections)} objects above confidence threshold!")

# 2. Display valid detections
for det in valid_detections:
    x, y, w, h, conf = det
    print(f"Object found at x={x:.2f}, y={y:.2f} with confidence {conf:.4f}")
    
rknn.release()