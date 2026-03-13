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

    # 3. Get Output Attributes
    # In rknn-toolkit-lite2, use get_output_info() to get quantization details
    output_info = rknn.get_output_info()
    
    # 4. Run inference
    img = Image.open(image_path).convert('RGB').resize((640, 640))
    input_data = np.expand_dims(np.array(img, dtype=np.uint8), 0)
    outputs = rknn.inference(inputs=[input_data])

    # 5. De-quantize
    for i, out in enumerate(outputs):
        # Access attributes directly from the info object
        scale = output_info[i].qnt_scale
        zp = output_info[i].qnt_zp
        
        # De-quantization: (int8_value - zp) * scale
        float_output = (out.astype(np.float32) - zp) * scale
        
        print(f"Output {i} shape: {float_output.shape}")
        print(f"Sample de-quantized values: {float_output.flatten()[:10]}")

    rknn.release()

if __name__ == "__main__":
    test_inference("YoloModels/v26/nano/NoNMS/model_test.rknn", 'Images/2.png')