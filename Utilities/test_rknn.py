import cv2
import numpy as np
from rknnlite.api import RKNNLite

# ── Change these to match your actual model and image ──────────────────────────
MODEL_PATH = "YoloModels/v26/nano/testing/color-3.1-v26-rk3588.rknn"
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

def section(title):
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")

# ── Load model ─────────────────────────────────────────────────────────────────
section("MODEL LOAD")
print(f"  Path: {MODEL_PATH}")
rknn = RKNNLite()
rknn.load_rknn(MODEL_PATH)
rknn.init_runtime(core_mask=RKNNLite.NPU_CORE_0_1_2)
print("  OK")

# ── Preprocess ─────────────────────────────────────────────────────────────────
section("PREPROCESSING")
img_bgr = cv2.imread(IMAGE_PATH)
assert img_bgr is not None, f"Failed to read image: {IMAGE_PATH}"
print(f"  Original image shape: {img_bgr.shape}")
img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
img_lb, scale, pad_x, pad_y = letterbox(img_rgb, INPUT_SIZE)
img_input = np.expand_dims(img_lb, axis=0).astype(np.uint8)
print(f"  Letterbox scale: {scale:.4f}  pad_x: {pad_x}  pad_y: {pad_y}")
print(f"  Input shape: {img_input.shape}  dtype: {img_input.dtype}")
print(f"  Input pixel range: {img_input.min()} – {img_input.max()}")

# ── Inference ──────────────────────────────────────────────────────────────────
section("RAW INFERENCE OUTPUT")
outputs = rknn.inference(inputs=[img_input])
print(f"  Number of output tensors: {len(outputs)}")
for i, o in enumerate(outputs):
    print(f"  Output[{i}]: shape={o.shape}  dtype={o.dtype}  min={o.min()}  max={o.max()}")

raw = outputs[0]

# ── Dequantize ─────────────────────────────────────────────────────────────────
section("DEQUANTIZATION")
if raw.dtype == np.int8:
    print("  int8 detected — dequantizing with /128.0")
    frame_out = raw[0].astype(np.float32) / 128.0
elif raw.dtype == np.uint8:
    print("  uint8 detected — dequantizing with /255.0")
    frame_out = raw[0].astype(np.float32) / 255.0
else:
    print(f"  {raw.dtype} — no dequantization needed")
    frame_out = raw[0].astype(np.float32)

# ── Shape / format detection ───────────────────────────────────────────────────
section("SHAPE & FORMAT DETECTION")
print(f"  Shape before transpose: {frame_out.shape}")
if frame_out.shape[0] == 5 and frame_out.shape[1] > 5:
    print("  Transposing: (5, N) → (N, 5)")
    frame_out = frame_out.T
elif frame_out.ndim == 2 and frame_out.shape[1] == 5:
    print("  Already (N, 5) — no transpose needed")
elif frame_out.ndim == 2 and frame_out.shape[1] == 6:
    print("  Shape is (N, 6) — this looks like an end2end/NMS output (x1,y1,x2,y2,conf,cls)")
    print("  This script assumes no-NMS (x_c,y_c,w,h,conf) — results below may be wrong")
else:
    print(f"  Unexpected shape {frame_out.shape} — proceeding anyway, inspect carefully")
print(f"  Shape after transpose: {frame_out.shape}")

# ── Column stats ───────────────────────────────────────────────────────────────
section("COLUMN STATS (x_c, y_c, w, h, conf_raw)")
num_cols = frame_out.shape[1]
labels = ['x_c ', 'y_c ', 'w   ', 'h   ', 'conf'] + [f'col{i}' for i in range(5, num_cols)]
for i in range(num_cols):
    col = frame_out[:, i]
    label = labels[i] if i < len(labels) else f'col{i}'
    print(f"  {label}: min={col.min():.4f}  max={col.max():.4f}  mean={col.mean():.4f}  "
          f"nonzero={np.count_nonzero(col)}  std={col.std():.4f}")

# ── Confidence: raw vs sigmoid vs treating as already-sigmoid ─────────────────
section("CONFIDENCE INTERPRETATION COMPARISON")
raw_conf = frame_out[:, 4]

sig_conf   = sigmoid(raw_conf)
# If the model already outputs probabilities (0-1) without sigmoid needed:
direct_conf = np.clip(raw_conf, 0, 1)

print(f"\n  [A] Sigmoid applied (use this if raw values span roughly -10 to +10):")
print(f"      min={sig_conf.min():.4f}  max={sig_conf.max():.4f}  mean={sig_conf.mean():.4f}")
print(f"      Above 0.1: {(sig_conf > 0.1).sum()}")
print(f"      Above 0.3: {(sig_conf > 0.3).sum()}")
print(f"      Above 0.5: {(sig_conf > 0.5).sum()}")
print(f"      Above 0.7: {(sig_conf > 0.7).sum()}")

print(f"\n  [B] Direct (no sigmoid) — use if raw values are already 0.0–1.0:")
print(f"      raw range: min={raw_conf.min():.4f}  max={raw_conf.max():.4f}")
print(f"      Above 0.1: {(direct_conf > 0.1).sum()}")
print(f"      Above 0.3: {(direct_conf > 0.3).sum()}")
print(f"      Above 0.5: {(direct_conf > 0.5).sum()}")
print(f"      Above 0.7: {(direct_conf > 0.7).sum()}")

print(f"\n  >>> If [A] and [B] give VERY different counts, you're using the wrong one.")
print(f"  >>> If raw_conf is mostly in -10..+10 range → use sigmoid (A).")
print(f"  >>> If raw_conf is mostly in 0..1 range → use direct (B).")

# ── Distribution of raw conf values ───────────────────────────────────────────
section("RAW CONF VALUE DISTRIBUTION (histogram buckets)")
buckets = [(-100,-10), (-10,-5), (-5,-1), (-1,0), (0,1), (1,5), (5,10), (10,100)]
for lo, hi in buckets:
    count = ((raw_conf >= lo) & (raw_conf < hi)).sum()
    bar = '█' * min(count // max(len(raw_conf)//50, 1), 40)
    print(f"  [{lo:>6}, {hi:>4}): {count:>6}  {bar}")

# ── Top 10 detections ──────────────────────────────────────────────────────────
section("TOP 10 ANCHORS BY SIGMOID CONFIDENCE")
top_idx = np.argsort(sig_conf)[::-1][:10]
print(f"  {'#':<3} {'x_c':>7} {'y_c':>7} {'w':>7} {'h':>7} {'raw_conf':>10} {'sigmoid':>8}  notes")
for i, idx in enumerate(top_idx):
    row = frame_out[idx]
    notes = ""
    # Check if box coordinates look sane for the input size
    tw, th = INPUT_SIZE
    if row[0] < 0 or row[0] > tw or row[1] < 0 or row[1] > th:
        notes += " ⚠ coords out of range"
    if row[2] <= 0 or row[3] <= 0:
        notes += " ⚠ zero/negative size"
    print(f"  {i:<3} {row[0]:>7.1f} {row[1]:>7.1f} {row[2]:>7.1f} {row[3]:>7.1f} "
          f"{row[4]:>10.4f} {sig_conf[idx]:>8.4f} {notes}")

# ── Visualize detections using sigmoid ────────────────────────────────────────
section(f"DETECTION ATTEMPT (sigmoid conf >= {CONF_THRESHOLD})")
orig_h, orig_w = img_bgr.shape[:2]
tw, th = INPUT_SIZE
s = min(tw / orig_w, th / orig_h)
nw, nh = int(orig_w * s), int(orig_h * s)
px = (tw - nw) / 2
py = (th - nh) / 2

mask = sig_conf >= CONF_THRESHOLD
detected = frame_out[mask]
detected_confs = sig_conf[mask]
print(f"  Detections: {len(detected)}")

if len(detected) > 0:
    vis = img_bgr.copy()
    for row, conf in zip(detected, detected_confs):
        x = (row[0] - px) / s
        y = (row[1] - py) / s
        w = row[2] / s
        h = row[3] / s
        x1, y1 = int(x - w/2), int(y - h/2)
        x2, y2 = int(x + w/2), int(y + h/2)
        cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(vis, f"{conf:.2f}", (x1, max(y1-5, 10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
    cv2.imwrite("debug_sigmoid.png", vis)
    print("  Saved: debug_sigmoid.png")
else:
    print("  No detections at this threshold with sigmoid.")
    print("  Saving image of top-3 anchors regardless for visual sanity check...")
    vis = img_bgr.copy()
    for idx in top_idx[:3]:
        row = frame_out[idx]
        x = (row[0] - px) / s
        y = (row[1] - py) / s
        w = row[2] / s
        h = row[3] / s
        x1, y1 = int(x - w/2), int(y - h/2)
        x2, y2 = int(x + w/2), int(y + h/2)
        cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 0, 255), 2)
        cv2.putText(vis, f"sig={sig_conf[idx]:.3f}", (x1, max(y1-5, 10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)
    cv2.imwrite("debug_top3_anchors.png", vis)
    print("  Saved: debug_top3_anchors.png  (red = low confidence, just for inspection)")

rknn.release()
print("\nDone.")