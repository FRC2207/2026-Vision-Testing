from rknn.api import RKNN

rknn = RKNN()

rknn.config(
    target_platform='rk3588',
    mean_values=[0,0,0],
    std_values=[1,1,1],
    batch_size=2, # So that both cameras can run effeiceintly in the same model.predict() call
    disable_rules=['fuse_exmatmul_add_mul_exsoftmax13_exmatmul_to_sdpa']
)

rknn.load_onnx(model=r'YoloModels/v26/nano/clean_rknns/color-3.1-v26.onnx')

rknn.build(
    do_quantization=True,
    dataset='Images/RknnDataset'
)

rknn.export_rknn('model.rknn')
rknn.release()