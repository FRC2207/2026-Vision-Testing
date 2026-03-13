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

rknn.release()