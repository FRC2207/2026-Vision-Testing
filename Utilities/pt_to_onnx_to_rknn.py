import os
import logging
from ultralytics import YOLO
from rknn.api import RKNN

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

MODELS = [
    ("YoloModels/v8_or_v11/gray-3.0.pt", None)
]

INPUT_SIZE = (640, 640)
TARGET_PLATFORM = "rk3588"
QUANTIZE = True
DATASET_TXT = "Images/RknnDataset_grayscale/dataset.txt"

def export_pt_to_onnx(pt_path: str, input_size: tuple) -> str:
    logger.info(f"Exporting {pt_path} → ONNX …")
    model = YOLO(pt_path)
    model.export(format="onnx", imgsz=input_size, simplify=True)

    pt_dir = os.path.dirname(os.path.abspath(pt_path))
    pt_basename = os.path.splitext(os.path.basename(pt_path))[0]
    onnx_path = os.path.join(pt_dir, f"{pt_basename}.onnx")

    if not os.path.exists(onnx_path):
        raise FileNotFoundError(f"Expected ONNX at {onnx_path} but it was not created.")

    logger.info(f"ONNX saved to {onnx_path}")
    return onnx_path

def convert_onnx_to_rknn(
    onnx_path: str,
    rknn_out_path: str,
    target_platform: str,
    quantize: bool,
    dataset_txt: str,
    batch_size: int,
) -> None:
    logger.info(f"Converting {onnx_path} → {rknn_out_path} (batch_size={batch_size}) …")

    rknn = RKNN()
    try:
        rknn.config(
            target_platform=target_platform,
            mean_values=[[0, 0, 0]],
            std_values=[[255, 255, 255]]
        )

        ret = rknn.load_onnx(model=onnx_path)
        if ret != 0:
            raise RuntimeError(f"load_onnx failed (ret={ret})")

        ret = rknn.build(do_quantization=quantize, dataset=dataset_txt if quantize else None)
        if ret != 0:
            raise RuntimeError(f"build failed (ret={ret})")

        ret = rknn.export_rknn(rknn_out_path)
        if ret != 0:
            raise RuntimeError(f"export_rknn failed (ret={ret})")

        logger.info(f"RKNN model saved to {rknn_out_path}")
    finally:
        rknn.release()


def resolve_rknn_path(pt_path: str, override: str | None) -> str:
    if override:
        return override
    pt_dir      = os.path.dirname(os.path.abspath(pt_path))
    pt_basename = os.path.splitext(os.path.basename(pt_path))[0]
    return os.path.join(pt_dir, f"{pt_basename}.rknn")

def main():
    successes, failures = [], []

    for pt_path, rknn_out_override in MODELS:
        rknn_out = resolve_rknn_path(pt_path, rknn_out_override)
        try:
            onnx_path = export_pt_to_onnx(pt_path, INPUT_SIZE)
            convert_onnx_to_rknn(
                onnx_path=onnx_path,
                rknn_out_path=rknn_out,
                target_platform=TARGET_PLATFORM,
                quantize=QUANTIZE,
                dataset_txt=DATASET_TXT,
                batch_size=1
            )
            successes.append(pt_path)
        except Exception as e:
            logger.error(f"Failed to convert {pt_path}: {e}")
            failures.append((pt_path, str(e)))

    print("\nBatch summary:")
    for p in successes:
        print(f"  {p}")
    for p, err in failures:
        print(f"  {p}  ->  {err}")
    print(f"  {len(successes)} succeeded, {len(failures)} failed")

if __name__ == "__main__":
    main()