import numpy as np
from rknnlite.api import RKNNLite
from PIL import Image

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def test_raw_output_to_boxes(model_path, image_path):
    rknn = RKNNLite()
    rknn.load_rknn(model_path)
    rknn.init_runtime(core_mask=RKNNLite.NPU_CORE_0)

    # 1. Run Inference
    img = Image.open(image_path).convert('RGB').resize((640, 640))
    input_data = np.expand_dims(np.array(img), 0)
    outputs = rknn.inference(inputs=[input_data])

    # 2. Reshape from (1, 5, 8400) to (8400, 5)
    # The columns are [xc, yc, w, h, conf]
    data = outputs[0][0].transpose()

    # 3. Apply Sigmoid to the confidence column (index 4)
    data[:, 4] = sigmoid(data[:, 4])

    # 4. Filter by confidence threshold (e.g., 0.25)
    conf_thresh = 0.25
    mask = data[:, 4] > conf_thresh
    filtered_data = data[mask]

    print(f"Detections found above {conf_thresh}: {len(filtered_data)}")

    # 5. Convert Center (xc, yc, w, h) to (x1, y1, x2, y2)
    # The math: x1 = xc - w/2, y1 = yc - h/2, etc.
    if len(filtered_data) > 0:
        xc, yc, w, h, conf = filtered_data.T
        x1 = xc - (w / 2)
        y1 = yc - (h / 2)
        x2 = xc + (w / 2)
        y2 = yc + (h / 2)

        for i in range(len(filtered_data)):
            print(f"Conf: {conf[i]:.2f} | Box: [{x1[i]:.1f}, {y1[i]:.1f}, {x2[i]:.1f}, {y2[i]:.1f}]")
    
    rknn.release()

if __name__ == "__main__":
    test_raw_output_to_boxes("model_test.rknn", 'Images/2.png')