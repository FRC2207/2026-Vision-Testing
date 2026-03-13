import numpy as np
from rknn.api import RKNN
import os
from PIL import Image

def test_inference(rknn_model_path, image_path):
    rknn = RKNN(verbose=False)
    
    print(f"Loading model: {rknn_model_path}")
    ret = rknn.load_rknn(rknn_model_path)
    if ret != 0:
        print("Failed to load model!")
        return

    print("Initializing Runtime")
    ret = rknn.init_runtime(target='rk3588')
    if ret != 0:
        print("Failed to init runtime!")
        return
    
    img = Image.open(image_path).convert('RGB')
    img = img.resize((640, 640))
    img_data = np.array(img, dtype=np.uint8)

    input_data = np.expand_dims(img_data, 0)

    print("Running inference")
    outputs = rknn.inference(inputs=[input_data])

    print(f"Output type: {type(outputs)}")
    print(f"Number of outputs: {len(outputs)}")
    print(f"Shape of first output: {outputs[0].shape}")
    print(f"First 10 values of output: {outputs[0].flatten()[:10]}")

    rknn.release()

if __name__ == "__main__":
    test_inference("YoloModels/v26/nano/NoNMS/model_test.rknn", 'Images/2.png')