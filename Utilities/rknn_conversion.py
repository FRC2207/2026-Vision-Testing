from rknn.api import RKNN

rknn = RKNN()

rknn.config(
    target_platform='rk3588',
    mean_values=[0,0,0],
    std_values=[1,1,1],
    dynamic_input=[[3, 640, 640]]
)

rknn.load_onnx(model=rf'YoloModels/v26/nano/clean_rknns/color-3.1-v26.onnx')

rknn.build(
    do_quantization=True, # Set True for INT8 (used for speed), False for FP16 (used fo accuracy)
    dataset='Images\RknnDataset'
)

rknn.export_rknn('model.rknn')
rknn.release()
