from rknn.api import RKNN

rknn = RKNN()

rknn.config(
    target_platform='rk3588',
    mean_values=[[0,0,0]],
    std_values=[[255,255,255]],
    # disable_rules=['fuse_exmatmul_add_mul_exsoftmax13_exmatmul_to_sdpa']
)

rknn.load_onnx(
    model='YoloModels/v26/nano/NoNMS/color-3.1-v26.onnx',
    input_size_list=[[1,640,640,3]]
)

rknn.build(
    do_quantization=True,
    dataset='Images/RknnDataset/dataset.txt'
)

rknn.export_rknn('model_test.rknn')

rknn.release()