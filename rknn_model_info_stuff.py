from rknn.api import RKNN

rknn = RKNN()
ret = rknn.load_rknn("YoloModels/v26/nano/color-3.1-v26.rknn")
if ret != 0:
    print("Failed to load RKNN model")
else:
    print("Input info:", rknn.get_input_output_info())
    print("Model quantization info:", rknn.configs)