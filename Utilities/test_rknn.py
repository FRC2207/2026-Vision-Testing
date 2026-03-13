import cv2
import numpy as np
from rknnlite.api import RKNNLite


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


# -------------------------------
# YOLO Letterbox (Correct Resize)
# -------------------------------
def letterbox(im, new_shape=(640, 640), color=(114,114,114)):
    shape = im.shape[:2]  # current shape [height, width]

    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])

    new_unpad = (int(round(shape[1] * r)), int(round(shape[0] * r)))

    dw = new_shape[1] - new_unpad[0]
    dh = new_shape[0] - new_unpad[1]

    dw /= 2
    dh /= 2

    im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)

    im = cv2.copyMakeBorder(
        im,
        int(round(dh - 0.1)),
        int(round(dh + 0.1)),
        int(round(dw - 0.1)),
        int(round(dw + 0.1)),
        cv2.BORDER_CONSTANT,
        value=color
    )

    return im, r, dw, dh


# -------------------------------
# RKNN Setup
# -------------------------------
rknn = RKNNLite()

print("Loading RKNN model...")
rknn.load_rknn('model_test_not_quant.rknn')

print("Init runtime...")
rknn.init_runtime(core_mask=RKNNLite.NPU_CORE_0)


# -------------------------------
# Load Image
# -------------------------------
img = cv2.imread('Images/1.png')
original_img = img.copy()

img_resized, r, dw, dh = letterbox(img, (640,640))

# normalize if model expects it
img_resized = img_resized.astype(np.float32) / 255.0

input_data = np.expand_dims(img_resized, axis=0)


# -------------------------------
# Inference
# -------------------------------
outputs = rknn.inference(inputs=[input_data])

print("Output shapes:", [o.shape for o in outputs])

raw = outputs[0][0]      # (5,8400)
data = raw.T             # (8400,5)

print("Total predictions:", len(data))


# -------------------------------
# Decode predictions
# -------------------------------
boxes = []
scores = []

for i in range(len(data)):

    x, y, w, h, conf = data[i]

    conf = sigmoid(conf)

    if conf < 0.4:
        continue

    # convert center → corners
    x1 = x - w/2
    y1 = y - h/2
    x2 = x + w/2
    y2 = y + h/2

    # undo letterbox
    x1 = (x1 - dw) / r
    y1 = (y1 - dh) / r
    x2 = (x2 - dw) / r
    y2 = (y2 - dh) / r

    x1 = int(max(0, x1))
    y1 = int(max(0, y1))
    x2 = int(min(original_img.shape[1], x2))
    y2 = int(min(original_img.shape[0], y2))

    boxes.append([x1, y1, x2-x1, y2-y1])
    scores.append(float(conf))


print("Candidates:", len(boxes))


# -------------------------------
# Apply NMS
# -------------------------------
indices = cv2.dnn.NMSBoxes(
    boxes,
    scores,
    score_threshold=0.4,
    nms_threshold=0.45
)

print("Final detections:", len(indices))


# -------------------------------
# Draw detections
# -------------------------------
for i in indices:

    i = int(i)

    x, y, w, h = boxes[i]

    cv2.rectangle(
        original_img,
        (x, y),
        (x+w, y+h),
        (0,255,0),
        2
    )

    label = f"{scores[i]:.2f}"

    cv2.putText(
        original_img,
        label,
        (x, y-5),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (0,255,0),
        2
    )

    print(f"Detection: {x,y,w,h} Conf={scores[i]:.2f}")


# -------------------------------
# Save result
# -------------------------------
cv2.imwrite("detected.png", original_img)

print("Saved detected.png")


# -------------------------------
# Cleanup
# -------------------------------
rknn.release()