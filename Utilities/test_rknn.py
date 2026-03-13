import numpy as np
from rknn.api import RKNN
import cv2

# --- SETTINGS ---
RKNN_MODEL = 'model_test_backup.rknn'
IMG_PATH = 'Images/1.png' # Use a real image of a game piece!

rknn = RKNN()
print('--> Loading model')
rknn.load_rknn(RKNN_MODEL)
print('--> Init runtime')
rknn.init_runtime()

# Prepare image
img = cv2.imread(IMG_PATH)
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
img = cv2.resize(img, (640, 640)) 

# Inference
outputs = rknn.inference(inputs=[img])

print("\n" + "="*30)
print("RKNN OUTPUT DIAGNOSTICS")
print("="*30)
for i, out in enumerate(outputs):
    print(f"Output [{i}] Shape: {out.shape}")
    # Flatten to see the first few values clearly
    flat = out.flatten()
    print(f"First 10 raw values: {flat[:10]}")
    print(f"Value Range: Min={flat.min():.4f}, Max={flat.max():.4f}")
print("="*30)

rknn.release()