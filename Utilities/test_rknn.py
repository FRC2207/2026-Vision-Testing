import numpy as np
from rknnlite.api import RKNNLite
from PIL import Image

def test_inference(rknn_model_path, image_path):
    rknn = RKNNLite()
    
    # 1. Load model
    print(f"--> Loading model: {rknn_model_path}")
    if rknn.load_rknn(rknn_model_path) != 0:
        print("Failed to load model!"); return

    # 2. Init runtime
    if rknn.init_runtime(core_mask=RKNNLite.NPU_CORE_0) != 0:
        print("Failed to init runtime!"); return

    # 3. Query output attributes (The correct way for RKNNLite)
    # This returns a list of objects containing scale and zp
    output_attrs = rknn.query(query=RKNNLite.RKNN_QUERY_OUTPUT_ATTR, index=None)
    
    # 4. Prepare image
    img = Image.open(image_path).convert('RGB').resize((640, 640))
    input_data = np.expand_dims(np.array(img, dtype=np.uint8), 0)

    # 5. Run inference
    outputs = rknn.inference(inputs=[input_data])

    # 6. De-quantize outputs
    for i, out in enumerate(outputs):
        # Access scale and zp from the queried attributes
        scale = output_attrs[i].qnt_scale
        zp = output_attrs[i].qnt_zp
        
        # De-quantize
        float_output = (out.astype(np.float32) - zp) * scale
        
        print(f"Output {i} shape: {float_output.shape}")
        print(f"Sample de-quantized values: {float_output.flatten()[:10]}")

    rknn.release()

if __name__ == "__main__":
    test_inference("YoloModels/v26/nano/NoNMS/model_test.rknn", 'Images/2.png')