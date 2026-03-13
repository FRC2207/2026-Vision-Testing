import cv2
import numpy as np
from rknnlite.api import RKNNLite

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# 1. Setup
rknn = RKNNLite()
rknn.load_rknn('model_test_not_quant.rknn')
rknn.init_runtime(core_mask=RKNNLite.NPU_CORE_0)

# 2. Preprocess
img = cv2.imread('Images/1.png')
original_img = img.copy()
img_resized = cv2.resize(img, (640, 640))
input_data = np.expand_dims(img_resized, axis=0)

# 3. Inference
outputs = rknn.inference(inputs=[input_data])
raw_data = outputs[0][0] # Shape (5, 8400)
data = raw_data.T        # Shape (8400, 5)

# 4. Decode and Filter
# Column 4 is confidence. Apply sigmoid if raw logits
confidences = sigmoid(data[:, 4])
boxes = data[confidences > 0.5] 
conf_scores = confidences[confidences > 0.5]

print(f"Found {len(boxes)} potential objects!")

# 5. Visualize
for i, box in enumerate(boxes):
    x, y, w, h = box[:4]
    # Assuming coordinates are already absolute for 640x640
    # Convert center_x, center_y, w, h to x1, y1, x2, y2
    x1 = int(x - w / 2)
    y1 = int(y - h / 2)
    x2 = int(x + w / 2)
    y2 = int(y + h / 2)
    
    cv2.rectangle(original_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
    print(f"Object {i}: {x1, y1, x2, y2} Conf: {conf_scores[i]:.2f}")
# 1. Prepare data for NMS (OpenCV requires list format)
boxes = []
confidences = []

# Assuming 'data' is your (8400, 5) array: [x, y, w, h, conf]
for i in range(len(data)):
    x, y, w, h, conf = data[i]
    if conf > 0.4: # Slightly lower threshold to catch candidates
        # Convert to top-left format for OpenCV
        x1 = int(x - w / 2)
        y1 = int(y - h / 2)
        boxes.append([x1, y1, int(w), int(h)])
        confidences.append(float(conf))

# 2. Apply NMS
# nms_threshold (e.g., 0.4) controls how much overlap is allowed
indices = cv2.dnn.NMSBoxes(boxes, confidences, score_threshold=0.4, nms_threshold=0.4)

# 3. Print clean results
print(f"Final detections after NMS: {len(indices)}")
for i in indices:
    box = boxes[i]
    print(f"Detection {i}: {box} Conf: {confidences[i]:.2f}")
    
cv2.imwrite('detected_1.png', original_img)
rknn.release()