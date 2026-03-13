import numpy as np
from rknnlite.api import RKNNLite
from PIL import Image

def test_inference(rknn_model_path, image_path):
    rknn = RKNNLite()
    
    # 1. Load model
    print(f"--> Loading model: {rknn_model_path}")
    if rknn.load_rknn(rknn_model_path) != 0:
        print("Failed to load model!"); return

    # 2. Init runtime (NPU_CORE_0 is usually standard for RK3588)
    if rknn.init_runtime(core_mask=RKNNLite.NPU_CORE_0) != 0:
        print("Failed to init runtime!"); return

    # 3. Get quantization parameters (Crucial for INT8)
    # Query output attributes to get scale and zero_point
    output_attrs = rknn.list_outputs() # Returns list of dictionaries with metadata
    
    # 4. Prepare image
    img = Image.open(image_path).convert('RGB').resize((640, 640))
    input_data = np.expand_dims(np.array(img, dtype=np.uint8), 0)

    # 5. Run inference
    outputs = rknn.inference(inputs=[input_data])

    # 6. De-quantize outputs
    # RKNNLite.list_outputs() usually provides the 'scale' and 'zp' (zero_point)
    # We iterate through outputs to convert them back to float32
    for i, out in enumerate(outputs):
        scale = output_attrs[i]['scale']
        zp = output_attrs[i]['zp']
        
        # De-quantization formula: (int8 - zp) * scale
        # Cast to float32 first to avoid overflow
        float_output = (out.astype(np.float32) - zp) * scale
        
        print(f"Output {i} shape: {float_output.shape}")
        print(f"Sample de-quantized values: {float_output.flatten()[:10]}")

    rknn.release()

if __name__ == "__main__":
    test_inference("YoloModels/v26/nano/NoNMS/model_test.rknn", 'Images/2.png')