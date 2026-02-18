import cv2
import math
import numpy as np
from ultralytics import YOLO
import matplotlib.pyplot as plt
import json

BALL_DIAMETER_INCHES = 6
CAMERA_FOV_DEGREES = 60
KNOWN_DISTANCE_INCHES = 24
KNOWN_PIXEL_HEIGHT = 82
IMAGE_PATH = "Images/1.png"
GRAYSCALE_MODEL = False
MIN_CONFIDENCE = 0.5
MARGIN = 10

focal_length_pixels = (KNOWN_PIXEL_HEIGHT * KNOWN_DISTANCE_INCHES) / BALL_DIAMETER_INCHES

model = YOLO("color-3.0.pt")
results = model(IMAGE_PATH)
r = results[0]

img = cv2.imread(IMAGE_PATH, cv2.IMREAD_GRAYSCALE)
img_h, img_w = img.shape[:2]

field_points = []
confidence_list = []

for i, box in enumerate(r.boxes.xyxy):
    x1, y1, x2, y2 = box.tolist()

    conf = float(r.boxes.conf[i].item())
    if conf < MIN_CONFIDENCE:
        continue
    if x1 < MARGIN or y1 < MARGIN or x2 > (img_w - MARGIN) or y2 > (img_h - MARGIN):
        continue
    confidence_list.append(conf)
    
    cx = (x1 + x2) / 2
    cy = (y1 + y2) / 2
    w_pixels = x2 - x1
    h_pixels = y2 - y1
    avg_pixels = (w_pixels + h_pixels) / 2
    distance = (BALL_DIAMETER_INCHES * focal_length_pixels) / avg_pixels
    
    dx_pixels = cx - (img_w / 2)
    angle_rad = math.radians(CAMERA_FOV_DEGREES) * (dx_pixels / img_w)
    
    X = math.tan(angle_rad) * distance
    Y = distance
    
    field_points.append([X, Y])
    
    cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
    cv2.circle(img, (int(cx), int(cy)), 5, (0, 0, 255), -1)

field_points = np.array(field_points)

img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

plt.figure(figsize=(14,7))

# Show original image with detections
plt.subplot(1,2,1)
plt.imshow(img_rgb)
plt.title("Detected Balls")
plt.axis('off')

# Show bird's-eye view
plt.subplot(1,2,2)
plt.scatter(field_points[:,0], field_points[:,1], c='red', s=100)
plt.xlim(-50, 50)
plt.ylim(0, 100)
plt.xlabel("Field X (inches)")
plt.ylabel("Field Y (inches)")
plt.title("Bird's Eye Fuel Map")
plt.grid(True)

plt.tight_layout()
plt.show()

output = {"points": [{"x": float(p[0]), "y": float(p[1])} for p in field_points]}

with open("ball_layout.json", "w") as f:
    json.dump(output, f, indent=4)

print(f"Saved {len(field_points)} balls to ball_layout.json. Average confidence: {np.mean(confidence_list):.2f}")
