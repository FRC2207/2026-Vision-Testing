import cv2
import numpy as np
from rknnlite.api import RKNNLite

def sigmoid(x):
    return 1 / (1 + np.exp(-x))


# -----------------
# Load RKNN
# -----------------
rknn = RKNNLite()
rknn.load_rknn("model_test_not_quant.rknn")
rknn.init_runtime(core_mask=RKNNLite.NPU_CORE_0)


# -----------------
# Load Image
# -----------------
img = cv2.imread("Images/1.png")
orig = img.copy()

img = cv2.resize(img, (640,640))
img = img.astype(np.float32) / 255.0
img = np.expand_dims(img,0)


# -----------------
# Inference
# -----------------
outputs = rknn.inference(inputs=[img])

raw = outputs[0][0]        # (5,8400)
data = raw.T               # (8400,5)

print("Predictions:",data.shape)


# -----------------
# Decode boxes
# -----------------
boxes = []
scores = []

for row in data:

    x,y,w,h,conf = row

    conf = sigmoid(conf)

    if conf < 0.4:
        continue

    x1 = int(x - w/2)
    y1 = int(y - h/2)
    x2 = int(x + w/2)
    y2 = int(y + h/2)

    boxes.append([x1,y1,x2-x1,y2-y1])
    scores.append(float(conf))


print("Candidates:",len(boxes))


# -----------------
# NMS
# -----------------
indices = cv2.dnn.NMSBoxes(
    boxes,
    scores,
    score_threshold=0.4,
    nms_threshold=0.45
)

print("Final:",len(indices))


# -----------------
# Draw
# -----------------
for i in indices:

    i=int(i)

    x,y,w,h = boxes[i]

    cv2.rectangle(orig,(x,y),(x+w,y+h),(0,255,0),2)

    cv2.putText(
        orig,
        f"{scores[i]:.2f}",
        (x,y-5),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (0,255,0),
        2
    )


cv2.imwrite("detected.png",orig)

rknn.release()