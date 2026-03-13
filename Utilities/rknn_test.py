from rknn.api import RKNN

rknn = RKNN()

rknn.config(
    target_platform='rk3588',
    mean_values=[[0,0,0]],
    std_values=[[255,255,255]],
    quantized_dtype='w8a8'
)

rknn.load_onnx(
    model='YoloModels/v26/nano/NoEnd2End/color-3.1-v26.onnx',
    outputs=["output0"]
)

rknn.build(
    do_quantization=True,
    dataset='Images/RknnDataset/dataset.txt'
)

print("Running accuracy analysis...")

rknn.accuracy_analysis(
    inputs=['Images/1.png'],
    target='rk3588',
    output_dir='./accuracy_analysis'
)

rknn.export_rknn('model_quant_debug.rknn')

rknn.release()