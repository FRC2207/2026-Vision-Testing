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
print(f"Inference output shape: {[o.shape for o in outputs]}")
raw_data = outputs[0][0] # Shape (5, 8400)
data = raw_data.T        # Shape (8400, 5)

# --- DEBUG: Print Raw Data for top 5 scores ---
# Sort by confidence (index 4)
sorted_indices = np.argsort(data[:, 4])[::-1]
print("\n--- TOP 5 RAW DATA (Verify Coordinates) ---")
for i in range(5):
    idx = sorted_indices[i]
    box = data[idx]
    print(f"Raw Output {i}: X={box[0]:.2f}, Y={box[1]:.2f}, W={box[2]:.2f}, H={box[3]:.2f}, Conf={box[4]:.2f}")

# --- Decode and Filter ---
# If your model was trained with Sigmoid, this is correct:
confidences = sigmoid(data[:, 4])
# If your model was trained with Softmax or is already activated, 
# you might not need sigmoid() at all. Try removing it if confidences are all 1.0 or 0.0.

# (Keep your existing NMS logic below)

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

# --- DEBUG: Comparative Decoding Test ---
print("\n--- COMPARATIVE COORDINATE TEST ---")
# Pick top index
top_idx = sorted_indices[0] 
raw_box = data[top_idx] # [x, y, w, h] (raw values)

# Strategy A: Standard YOLO (Center-Point)
x1_a = int(raw_box[0] - raw_box[2] / 2)
y1_a = int(raw_box[1] - raw_box[3] / 2)
x2_a = int(raw_box[0] + raw_box[2] / 2)
y2_a = int(raw_box[1] + raw_box[3] / 2)

# Strategy B: Corner-Points (Top-Left, Bottom-Right)
x1_b = int(raw_box[0])
y1_b = int(raw_box[1])
x2_b = int(raw_box[2])
y2_b = int(raw_box[3])

# Strategy C: Offset-from-Grid (Sometimes used in custom models)
# This assumes x,y are grid offsets, and w,h are log-scale multipliers
# (Common if output seems to be clustered in one corner)

print(f"Raw NPU Values: {raw_box[:4]}")
print(f"Strategy A (Center-based): [{x1_a}, {y1_a}, {x2_a}, {y2_a}]")
print(f"Strategy B (Corner-based):  [{x1_b}, {y1_b}, {x2_b}, {y2_b}]")

# Define decoding strategies
# Each tuple: (scale_factor, use_center_math)
strategies = [
    (1.0, True),   # A: 640x640 absolute (Standard)
    (640.0, True), # B: Normalized 0-1 (e.g. YOLOv8 format)
    (1.0, False),  # C: Absolute Corners (No w/2 calculation)
    (640.0, False) # D: Normalized Corners
]

for idx, (scale, use_center) in enumerate(strategies):
    temp_img = img.copy()
    print(f"\n--- Generating Gallery {idx} | Scale: {scale} | CenterMath: {use_center} ---")
    
    boxes_for_nms = []
    scores_for_nms = []
    
    for i in range(len(data)):
        x, y, w, h, conf = data[i]
        
        # Apply scaling
        x, y, w, h = np.nan_to_num([x, y, w, h], nan=0.0, posinf=640.0, neginf=0.0)
        
        if use_center:
            x1, y1 = int(x - w / 2), int(y - h / 2)
            x2, y2 = int(x + w / 2), int(y + h / 2)
        else:
            x1, y1 = int(x), int(y)
            x2, y2 = int(w), int(h) # In corner format, w/h are actually x2/y2
            
        if conf > 0.4:
            cv2.rectangle(temp_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            boxes_for_nms.append([x1, y1, x2-x1, y2-y1])
            scores_for_nms.append(float(conf))
            
    cv2.imwrite(f'detected_strategy_{idx}.png', temp_img)
    print(f"Saved: detected_strategy_{idx}.png")
rknn.release()