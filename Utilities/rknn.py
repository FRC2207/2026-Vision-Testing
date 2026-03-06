from rknn.api import RKNN

rknn = RKNN()

rknn.config(
    target_platform='rk3588',
)

rknn.load_onnx(model=rf'YoloModels\v26\nano\clean_rknns\color-3.1-v26.onnx')

rknn.build(
    do_quantization=True, # Set True for INT8 (Speed), False for FP16 (Accuracy)
    dataset='Images\RknnDataset'
)

rknn.export_rknn('model.rknn')
rknn.release()
