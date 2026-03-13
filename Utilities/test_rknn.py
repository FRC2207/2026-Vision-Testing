import numpy as np
from rknnlite.api import RKNNLite
from PIL import Image

def test_inference(rknn_model_path, image_path):
    rknn = RKNNLite()
    
    # 1. Load model
    if rknn.load_rknn(rknn_model_path) != 0:
        print("Failed to load model!"); return

    # 2. Init runtime
    if rknn.init_runtime(core_mask=RKNNLite.NPU_CORE_0) != 0:
        print("Failed to init runtime!"); return

    # 3. Prepare image
    img = Image.open(image_path).convert('RGB').resize((640, 640))
    input_data = np.expand_dims(np.array(img, dtype=np.uint8), 0)

    # 4. Inference
    print("Running inference...")
    outputs = rknn.inference(inputs=[input_data])

    # 5. Inspect the raw results
    for i, out in enumerate(outputs):
        print(f"Output {i} shape: {out.shape}")
        print(f"Output {i} dtype: {out.dtype}")
        print(f"Max value: {np.max(out)}")
        print(f"Min value: {np.min(out)}")
        print(f"First 10 raw values: {out.flatten()[:10]}")

    rknn.release()

if __name__ == "__main__":
    test_inference("YoloModels/v26/nano/NoNMS/model_test.rknn", 'Images/2.png')