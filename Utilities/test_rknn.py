import cv2
import numpy as np
from rknnlite.api import RKNNLite

# ── Change these to match your actual model and image ──────────────────────────
MODEL_PATH = "YoloModels/v8_or_v11/3.1-320x320/model.rknn"
IMAGE_PATH = "Images/1.png"
INPUT_SIZE = (320, 320)
CONF_THRESHOLD = 0.3
# ───────────────────────────────────────────────────────────────────────────────

def letterbox(img, target_size):
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

def sigmoid(x):
    return 1 / (1 + np.exp(-np.clip(x, -88, 88)))

# ── Load model ─────────────────────────────────────────────────────────────────
print(f"Loading model: {MODEL_PATH}")
rknn = RKNNLite()
rknn.load_rknn(MODEL_PATH)
rknn.init_runtime(core_mask=RKNNLite.NPU_CORE_0_1_2)

# ── Preprocess ─────────────────────────────────────────────────────────────────
img_bgr = cv2.imread(IMAGE_PATH)
assert img_bgr is not None, f"Failed to read image: {IMAGE_PATH}"
img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
img_lb, scale, pad_x, pad_y = letterbox(img_rgb, INPUT_SIZE)
img_input = np.expand_dims(img_lb, axis=0).astype(np.uint8)

print(f"Input shape: {img_input.shape}  dtype: {img_input.dtype}")
print(f"Input pixel range: {img_input.min()} – {img_input.max()}")

# ── Inference ──────────────────────────────────────────────────────────────────
outputs = rknn.inference(inputs=[img_input])
raw = outputs[0]
print(f"\nRaw output shape: {raw.shape}  dtype: {raw.dtype}")
print(f"Raw value range: {raw.min()} – {raw.max()}")

# ── Dequantize only if int8 ────────────────────────────────────────────────────
if raw.dtype == np.int8:
    print("Output is int8 — dequantizing with /128.0")
    frame_out = raw[0].astype(np.float32) / 128.0
else:
    print(f"Output is already {raw.dtype} — no dequantization needed")
    frame_out = raw[0].astype(np.float32)

# ── Transpose to [num_anchors, 5] ─────────────────────────────────────────────
if frame_out.shape[0] == 5 and frame_out.shape[1] > 5:
    frame_out = frame_out.T
print(f"After transpose: {frame_out.shape}")

# ── Column stats ───────────────────────────────────────────────────────────────
print(f"\n=== Column stats (x_c, y_c, w, h, conf_raw) ===")
labels = ['x_c ', 'y_c ', 'w   ', 'h   ', 'conf']
for i, label in enumerate(labels):
    col = frame_out[:, i]
    print(f"  {label}: min={col.min():.4f}  max={col.max():.4f}  mean={col.mean():.4f}  nonzero={np.count_nonzero(col)}")

# ── Confidence analysis ────────────────────────────────────────────────────────
raw_conf = frame_out[:, 4]
sig_conf = sigmoid(raw_conf)

print(f"\n=== Confidence after sigmoid ===")
print(f"  min={sig_conf.min():.4f}  max={sig_conf.max():.4f}  mean={sig_conf.mean():.4f}")
print(f"  Above 0.1: {(sig_conf > 0.1).sum()}")
print(f"  Above 0.3: {(sig_conf > 0.3).sum()}")
print(f"  Above 0.5: {(sig_conf > 0.5).sum()}")

print(f"\n=== Top 10 rows by sigmoid confidence ===")
top_idx = np.argsort(sig_conf)[::-1][:10]
for i, idx in enumerate(top_idx):
    row = frame_out[idx]
    print(f"  {i}: x_c={row[0]:.1f} y_c={row[1]:.1f} w={row[2]:.1f} h={row[3]:.1f} raw_conf={row[4]:.4f} sigmoid={sig_conf[idx]:.4f}")

# ── Try detecting with threshold ───────────────────────────────────────────────
mask = sig_conf >= CONF_THRESHOLD
detected = frame_out[mask]
print(f"\n=== Detections above {CONF_THRESHOLD} confidence: {len(detected)} ===")
if len(detected) > 0:
    orig_h, orig_w = img_bgr.shape[:2]
    tw, th = INPUT_SIZE
    s = min(tw / orig_w, th / orig_h)
    nw, nh = int(orig_w * s), int(orig_h * s)
    px = (tw - nw) / 2
    py = (th - nh) / 2

    vis = img_bgr.copy()
    for row in detected:
        x = (row[0] - px) / s
        y = (row[1] - py) / s
        w = row[2] / s
        h = row[3] / s
        x1, y1 = int(x - w/2), int(y - h/2)
        x2, y2 = int(x + w/2), int(y + h/2)
        cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 0), 2)

    out_path = "test_output.png"
    cv2.imwrite(out_path, vis)
    print(f"  Saved annotated image to: {out_path}")
else:
    print("  No detections — likely a quantization calibration problem.")
    print("  Try rebuilding the .rknn with do_quantization=False to confirm.")

rknn.release()