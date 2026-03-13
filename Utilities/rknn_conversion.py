from rknn.api import RKNN

rknn = RKNN()

rknn.config(
    target_platform='rk3588',
    mean_values=[[0,0,0]],
    std_values=[[255,255,255]],
    # disable_rules=['fuse_exmatmul_add_mul_exsoftmax13_exmatmul_to_sdpa'],
    quantized_dtype='w8a8',
    quantized_hybrid_level=2 
)

rknn.load_onnx(
    model='YoloModels/v26/nano/NoEnd2End/color-3.1-v26.onnx',
    outputs=["output0"]
)

rknn.build(
    do_quantization=True,
    dataset='Images/RknnDataset/dataset.txt'
)

rknn.export_rknn('model_test_hybrid_quant.rknn')

rknn.release()