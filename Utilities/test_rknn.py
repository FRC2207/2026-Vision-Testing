import numpy as np
from rknnlite.api import RKNNLite
from PIL import Image

def test_rknn_e2e(model_path, image_path):
    rknn = RKNNLite()
    if rknn.load_rknn(model_path) != 0: return
    if rknn.init_runtime(core_mask=RKNNLite.NPU_CORE_0) != 0: return

    # 1. Get Scaling Info (This is what fixes the huge numbers)
    # Since rknnlite is sparse, we will try to get it, or use a default 'guess'
    # if the query fails.
    try:
        output_attrs = rknn.query(query=RKNNLite.RKNN_QUERY_OUTPUT_ATTR)
        scale = output_attrs[0].qnt_scale
        zp = output_attrs[0].qnt_zp
    except:
        print("Could not query scale. Using raw data.")
        scale, zp = 1.0, 0

    # 2. Prepare Image
    img = Image.open(image_path).convert('RGB').resize((640, 640))
    input_data = np.expand_dims(np.array(img, dtype=np.uint8), 0)

    # 3. Inference
    outputs = rknn.inference(inputs=[input_data])
    
    # outputs[0] is (1, 300, 6) -> [x1, y1, x2, y2, score, class]
    detections = outputs[0][0]

    print("\n--- DE-QUANTIZED RESULTS ---")
    
    count = 0
    for i in range(300):
        # Apply the math: float = (int - zp) * scale
        raw_score = detections[i, 4]
        score = (raw_score - zp) * scale
        
        # Only print if it looks like a real detection
        if score > 0.1: # Your threshold
            x1, y1, x2, y2 = detections[i, 0:4]
            cls = int(detections[i, 5])
            print(f"Detect! Class: {cls} | Conf: {score:.3f} | Box: [{x1:.1f}, {y1:.1f}, {x2:.1f}, {y2:.1f}]")
            count += 1
    
    if count == 0:
        print("No detections found above 0.1 threshold.")

    rknn.release()

if __name__ == "__main__":
    test_rknn_e2e("YoloModels/v26/nano/NoNMS/model_test.rknn", 'Images/2.png')