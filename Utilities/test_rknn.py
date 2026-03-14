import cv2
import numpy as np
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

img_input = np.expand_dims(img_lb, axis=0).astype(np.uint8)

outputs = rknn.inference(inputs=[img_input])
raw = outputs[0] # (1, 5, 8400) int8
print(f"Raw output shape: {raw.shape}  dtype: {raw.dtype}")
print(f"Raw int8 min/max: {raw.min()}  {raw.max()}")

# Dequantize int8 → float32
frame_out = raw[0].astype(np.float32) / 128.0 # (5, 8400)

if frame_out.shape[0] == 5 and frame_out.shape[1] > 5:
    frame_out = frame_out.T # (8400, 5)

print(f"\nAfter dequant + transpose: {frame_out.shape}")
print(f"\n=== Column stats (x_c, y_c, w, h, conf) ===")
labels = ['x_c ', 'y_c ', 'w   ', 'h   ', 'conf']
for i, label in enumerate(labels):
    col = frame_out[:, i]
    print(f"  {label}: min={col.min():.4f}  max={col.max():.4f}  mean={col.mean():.4f}")

print("\n=== Top 10 rows by confidence ===")
sorted_rows = frame_out[np.argsort(frame_out[:, 4])[::-1]][:10]
for i, row in enumerate(sorted_rows):
    print(f"  {i}: x_c={row[0]:.1f} y_c={row[1]:.1f} w={row[2]:.1f} h={row[3]:.1f} conf={row[4]:.4f}")

print("\n=== First 5 raw rows (no sorting) ===")
for i, row in enumerate(frame_out[:5]):
    print(f"  {i}: {row}")

rknn.release()