from rknn.api import RKNN

rknn = RKNN()

rknn.config(
    target_platform='rk3588',
    input_size_list=[[1, 3, 640, 640]]
)

rknn.load_onnx(model=rf'YoloModels/v26/nano/clean_rknns/color-3.1-v26.onnx')

rknn.build(
    do_quantization=True, # Set True for INT8 (used for speed), False for FP16 (used fo accuracy)
    dataset='Images\RknnDataset'
)

rknn.export_rknn('model.rknn')
rknn.release()
