import argparse
import onnx
import onnx_graphsurgeon as gs
import numpy as np
# Note, this is a very BETA file, it will attempt to manually strip NMS from a onnx file, resulting in clean rknn quantization.

def strip_nms(input_path: str, output_path: str):
    print(f"Loading {input_path}...")
    model = onnx.load(input_path)
    graph = gs.import_onnx(model)

    print("Graph inputs:", [i.name for i in graph.inputs])
    print("Graph outputs:", [o.name for o in graph.outputs])
    print("Output shapes:", [o.shape for o in graph.outputs])

    raw_output_tensor = None
    best_candidate = None

    for tensor in graph.tensors().values():
        if not hasattr(tensor, 'shape') or tensor.shape is None:
            continue
        shape = tensor.shape
        if (
            len(shape) == 3
            and shape[0] == 1 # batch=1
            and shape[2] == 8400 # 8400 anchors
            and shape[1] >= 5 # at least 4 box coords + 1 conf
        ):
            print(f"  Found candidate raw tensor: {tensor.name}  shape={shape}")
            best_candidate = tensor

    if best_candidate is None:
        for tensor in graph.tensors().values():
            if not hasattr(tensor, 'shape') or tensor.shape is None:
                continue
            shape = tensor.shape
            if len(shape) == 3 and 8400 in shape:
                print(f"  Fallback candidate: {tensor.name}  shape={shape}")
                best_candidate = tensor
                break

    if best_candidate is None:
        raise RuntimeError(
            "Could not find raw detection tensor (1, nc+4, 8400).\n"
            "The model may already be NMS-free, or use a different anchor count.\n"
            "Open the ONNX in Netron (https://netron.app) and find the tensor\n"
            "just before the TopK node, then pass --output_node <tensor_name>."
        )

    raw_output_tensor = best_candidate
    print(f"\nUsing raw output tensor: {raw_output_tensor.name}  shape={raw_output_tensor.shape}")

    graph.outputs = [raw_output_tensor]

    graph.cleanup()
    graph.toposort()

    print(f"\nNew graph outputs: {[o.name for o in graph.outputs]}")
    print(f"New output shapes: {[o.shape for o in graph.outputs]}")

    model_out = gs.export_onnx(graph)

    model_out = onnx.shape_inference.infer_shapes(model_out)
    onnx.save(model_out, output_path)
    print(f"\nSaved stripped model to: {output_path}")
    print("You can verify in Netron — output should now be (1, 5, 8400) or (1, nc+4, 8400).")


if __name__ == "__main__":
    strip_nms("YoloModels/v26/nano/NoNMS/color-3.1-v26.onnx", "YoloModels/v26/nano/NoNMS/color-3.1-v26-NoNMS.onnx")