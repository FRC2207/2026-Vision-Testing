import cv2
import numpy as np
from rknnlite.api import RKNNLite


MODEL_PATH = "model_test_hybrid_quant.rknn"   # change between your models
IMAGE_PATH = "Images/1.png"


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def letterbox(img, size=(640,640)):
    h,w = img.shape[:2]
    tw,th = size

    scale = min(tw/w, th/h)
    nw,nh = int(w*scale), int(h*scale)

    resized = cv2.resize(img,(nw,nh))

    canvas = np.full((th,tw,3),114,dtype=np.uint8)

    x_offset = (tw-nw)//2
    y_offset = (th-nh)//2

    canvas[y_offset:y_offset+nh, x_offset:x_offset+nw] = resized

    return canvas


def print_stats(name, arr):

    arr = arr.astype(np.float32)

    print("\n====",name,"====")
    print("shape:",arr.shape)
    print("dtype:",arr.dtype)

    print("min:",arr.min())
    print("max:",arr.max())
    print("mean:",arr.mean())
    print("std:",arr.std())

    zero_count = np.sum(arr == 0)
    print("zeros:", zero_count, "/", arr.size)

    unique_vals = np.unique(arr[:1000])
    print("unique sample:", unique_vals[:20])


print("\nLOADING MODEL\n")

rknn = RKNNLite()
rknn.load_rknn(MODEL_PATH)
rknn.init_runtime()

print("\nMODEL IO INFO\n")

print("inputs:", rknn.get_input_tensor_info())
print("outputs:", rknn.get_output_tensor_info())


print("\nREAD IMAGE\n")

img = cv2.imread(IMAGE_PATH)
img = letterbox(img)

img = img.astype(np.float32) / 255.0
img = np.expand_dims(img,0)

print_stats("INPUT", img)


print("\nRUNNING INFERENCE\n")

outputs = rknn.inference(inputs=[img])

out = outputs[0]

print_stats("RAW OUTPUT", out)


print("\nRAW OUTPUT SAMPLE\n")

flat = out.reshape(-1)

print(flat[:50])


print("\nCHECK YOLO FORMAT\n")

data = out[0]

if data.shape[0] < data.shape[1]:
    data = data.T

print("decoded shape:", data.shape)

print("\nFIRST 10 DETECTIONS")
print(data[:10])


if data.shape[1] >= 5:

    conf = data[:,4]

    print_stats("CONF CHANNEL", conf)

    sig = sigmoid(conf)

    print_stats("SIGMOID CONF", sig)

    print("\nSIGMOID SAMPLE")
    print(sig[:20])


print("\nDONE\n")

rknn.release()