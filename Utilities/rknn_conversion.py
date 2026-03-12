from rknn.api import RKNN

rknn = RKNN()

rknn.config(
    target_platform='rk3588',
    mean_values=[0, 0, 0],
    std_values=[1, 1, 1],
    disable_rules=['fuse_exmatmul_add_mul_exsoftmax13_exmatmul_to_sdpa'],
    dynamic_batch=True
)

rknn.load_onnx(model=r'YoloModels/v26/nano/clean_rknn/color-3.1-v26.onnx')

rknn.build(
    do_quantization=True,
    dataset='Images/RknnDataset/dataset.txt'
)

rknn.export_rknn('model.rknn')

rknn.release()