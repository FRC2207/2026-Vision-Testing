from rknn.api import RKNN

rknn = RKNN()

rknn.config(
    target_platform='rk3588',
    mean_values=[[0, 0, 0]],
    std_values=[[255, 255, 255]],
    quantized_dtype='w8a8',
    quantized_algorithm='kl_divergence',
)

rknn.load_onnx(
    model='YoloModels/v26/nano/NoNMS/color-3.1-v26-nonms.onnx',
    outputs=None,
)

ret = rknn.hybrid_quantization_step1(
    dataset='Images/RknnDataset/dataset.txt',
    proposal=True, # Toolkit will suggest which layers to make float
    proposal_dataset_size=50
)

if ret != 0:
    print("Step 1 failed")
    exit()

rknn.release()
print("\nDone. Now edit model.quantization.cfg:")
print("  Find layers with '/model.23/' in their name (the detection head)")
print("  Change 'asymmetric_quantized-8' → 'float' for those layers")
print("  Then run rknn_conversion_step2.py")