import numpy as np
from rknnlite.api import RKNNLite
from PIL import Image

def test_raw_output(model_path, image_path):
    rknn = RKNNLite()

    print(f"--> Loading model: {model_path}")
    if rknn.load_rknn(model_path) != 0:
        print("Load failed!"); return

    print("--> Initializing Runtime...")
    if rknn.init_runtime(core_mask=RKNNLite.NPU_CORE_0) != 0:
        print("Init failed!"); return

    # 1. Load and Resize image (RGB)
    img = Image.open(image_path).convert('RGB').resize((640, 640))
    img_data = np.array(img, dtype=np.uint8)
    input_data = np.expand_dims(img_data, 0) # Shape (1, 640, 640, 3)

    # 2. Run Inference
    print("--> Running Inference...")
    outputs = rknn.inference(inputs=[input_data])

    # 3. Process the (1, 5, 8400) shape
    # outputs[0] is (1, 5, 8400)
    # We want it as (8400, 5) -> [x_c, y_c, w, h, conf]
    raw_results = outputs[0][0].transpose() 

    # 4. Extract Confidence
    confidences = raw_results[:, 4]
    max_conf = np.max(confidences)
    avg_conf = np.mean(confidences)
    
    print("\n--- RESULTS ---")
    print(f"Output Shape: {outputs[0].shape}")
    print(f"Max Confidence Found: {max_conf:.4f}")
    print(f"Average Confidence: {avg_conf:.6f}")

    # 5. Get the Top 3 "Best" Detections to verify data
    top_indices = np.argsort(confidences)[-3:][::-1]
    
    print("\nTop 3 Candidates:")
    for i in top_indices:
        det = raw_results[i]
        print(f"Conf: {det[4]:.4f} | Rect: [xc:{det[0]:.1f}, yc:{det[1]:.1f}, w:{det[2]:.1f}, h:{det[3]:.1f}]")

    if max_conf < 0.1:
        print("\nWARNING: Confidence is very low. Check your normalization (mean/std) in the RKNN build script.")
    else:
        print("\nSUCCESS: Model is generating detections!")

    rknn.release()

if __name__ == "__main__":
    # Update these paths to your actual files
    test_raw_output("YoloModels/v26/nano/NoNMS/model_test.rknn", 'Images/2.png')