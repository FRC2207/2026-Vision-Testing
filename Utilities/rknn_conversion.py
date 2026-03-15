from rknn.api import RKNN

rknn = RKNN()

rknn.config(
    target_platform='rk3588',
    mean_values=[[0,0,0]],
    std_values=[[255,255,255]],
    disable_rules=['fuse_exmatmul_add_mul_exsoftmax13_exmatmul_to_sdpa'],
    # quantized_algorithm='kl_divergence', # IDK what this does
    # quantized_dtype='w8a8',
    # quantized_hybrid_level=3
)

rknn.load_onnx(
    model='YoloModels/v8_or_v11/3.1-320x320/color-3.1-v11.onnx'
    # outputs=["output0"]
)

rknn.build(
    do_quantization=True,
    dataset='Images/RknnDataset/dataset.txt'
)

rknn.export_rknn('model.rknn')

rknn.release()