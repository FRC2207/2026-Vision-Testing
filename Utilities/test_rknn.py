"""
Debug script — prints raw output values so we can see what confidence
scores actually look like before any threshold is applied.
"""
import cv2
import numpy as np
import time
from rknnlite.api import RKNNLite

INPUT_SIZE = (640, 640)

def letterbox(img, target_size=(640, 640)):
    h, w = img.shape[:2]
    tw, th = target_size
    scale = min(tw / w, th / h)
    nw, nh = int(w * scale), int(h * scale)
    resized = cv2.resize(img, (nw, nh))
    pad_w, pad_h = tw - nw, th - nh
    top, left = pad_h // 2, pad_w // 2
    bottom, right = pad_h - top, pad_w - left
    padded = cv2.copyMakeBorder(
        resized, top, bottom, left, right,
        cv2.BORDER_CONSTANT, value=(114, 114, 114)
    )
    return padded, scale, left, top

rknn = RKNNLite()
rknn.load_rknn("YoloModels/v26/nano/test/model.rknn")
rknn.init_runtime(core_mask=RKNNLite.NPU_CORE_0_1_2)

img_bgr = cv2.imread("Images/1.png")
img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
img_lb, scale, pad_x, pad_y = letterbox(img_rgb, INPUT_SIZE)
img_input = np.expand_dims(img_lb, axis=0).astype(np.float16)

outputs = rknn.inference(inputs=[img_input])
raw = outputs[0][0]  # (300, 6)

print("=== Full output column stats ===")
print(f"x1  : min={raw[:,0].min():.3f}  max={raw[:,0].max():.3f}")
print(f"y1  : min={raw[:,1].min():.3f}  max={raw[:,1].max():.3f}")
print(f"x2  : min={raw[:,2].min():.3f}  max={raw[:,2].max():.3f}")
print(f"y2  : min={raw[:,3].min():.3f}  max={raw[:,3].max():.3f}")
print(f"conf: min={raw[:,4].min():.4f}  max={raw[:,4].max():.4f}  mean={raw[:,4].mean():.4f}")
print(f"cls : min={raw[:,5].min():.3f}  max={raw[:,5].max():.3f}")

print("\n=== Top 10 rows sorted by conf (col 4) ===")
sorted_rows = raw[np.argsort(raw[:, 4])[::-1]][:10]
for i, row in enumerate(sorted_rows):
    print(f"  {i}: x1={row[0]:.1f} y1={row[1]:.1f} x2={row[2]:.1f} y2={row[3]:.1f} conf={row[4]:.4f} cls={row[5]:.0f}")

print("\n=== Top 10 rows sorted by col 5 (in case conf/cls are swapped) ===")
sorted_rows2 = raw[np.argsort(raw[:, 5])[::-1]][:10]
for i, row in enumerate(sorted_rows2):
    print(f"  {i}: x1={row[0]:.1f} y1={row[1]:.1f} x2={row[2]:.1f} y2={row[3]:.1f} col4={row[4]:.4f} col5={row[5]:.4f}")

print("\n=== First 5 raw rows (no sorting) ===")
for i, row in enumerate(raw[:5]):
    print(f"  {i}: {row}")

rknn.release()