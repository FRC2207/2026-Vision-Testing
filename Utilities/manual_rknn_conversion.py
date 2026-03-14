from rknn.api import RKNN

rknn = RKNN()

ret = rknn.hybrid_quantization_step2(
    'color-3.1-v26-nonms.model', # model_input
    'color-3.1-v26-nonms.data', # data_input
    'color-3.1-v26-nonms.quantization.cfg', # model_quantization_cfg
)

if ret != 0:
    print("Step 2 failed")
    exit()

rknn.export_rknn('model.rknn')
rknn.release()