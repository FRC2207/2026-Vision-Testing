import numpy as np
from rknnlite.api import RKNNLite
from PIL import Image

rknn = RKNNLite()
rknn.load_rknn("model.rknn")
rknn.init_runtime()

img = Image.open('Images/2.png').convert('RGB').resize((640, 640))
outputs = rknn.inference(inputs=[np.expand_dims(np.array(img), 0)])

# Print the first 30 numbers in the entire output buffer
print("Raw Tensor Dump (First 30 values):")
print(outputs[0].flatten()[:30])

rknn.release()