import cv2
import json
import numpy as np
from ultralytics import YOLO
from pathlib import Path


def get_float_input(prompt: str, condition=lambda x: x > 0, error_msg="Invalid input.") -> float:
    while True:
        try:
            value = float(input(prompt))
            if condition(value):
                return value
            print(error_msg)
        except ValueError:
            print("Please enter a valid number.")


def draw_detections(frame: np.ndarray, results) -> tuple[np.ndarray, list]:
    """Draw bounding boxes on frame and return annotated frame + box list."""
    annotated = frame.copy()
    boxes = results.boxes.xyxy.cpu().numpy()
    confidences = results.boxes.conf.cpu().numpy()

    for i, (box, conf) in enumerate(zip(boxes, confidences)):
        x1, y1, x2, y2 = map(int, box)
        cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(annotated, f"{conf:.2f}", (x1, y1 - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    cv2.putText(annotated, f"Detections: {len(boxes)}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 200, 255), 2)
    cv2.putText(annotated, "SPACE: capture  |  Q: quit  |  C: clear captures", (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)

    return annotated, boxes


def get_best_detection(boxes: np.ndarray) -> int:
    """Return index of largest bounding box."""
    areas = [(x2 - x1) * (y2 - y1) for x1, y1, x2, y2 in boxes]
    return int(np.argmax(areas))


def calibrate_camera(camera_id: int = 0, yolo_model_path: str = "YoloModels/gray-3.0.pt", ball_diameter: float = 9.43):
    camera_fov = get_float_input(
        "\nEnter camera FOV in degrees (e.g., 60): ",
        condition=lambda x: 0 < x < 180,
        error_msg="Please enter a value between 0 and 180."
    )

    print("\nLoading YOLO model...")
    model = YOLO(yolo_model_path)

    cap = cv2.VideoCapture(camera_id)
    if not cap.isOpened():
        print(f"Error: Cannot open camera {camera_id}")
        return None

    print("\nCAMERA FEED ACTIVE — model running live")
    print("  SPACE : capture current frame")
    print("  C     : clear all captures")
    print("  Q     : quit\n")

    captures = []  # list of (frame, boxes, confidences)

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error reading frame.")
            break

        results = model(frame, verbose=False)[0]
        boxes = results.boxes.xyxy.cpu().numpy()
        confidences = results.boxes.conf.cpu().numpy()

        annotated, _ = draw_detections(frame, results)

        # Show capture count
        cv2.putText(annotated, f"Captures: {len(captures)}", (10, 90),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

        cv2.imshow("Camera Calibration", annotated)
        key = cv2.waitKey(1) & 0xFF

        if key == ord(' '):
            if len(boxes) == 0:
                print("  ✗ No detection in frame — try again.")
            else:
                captures.append((frame.copy(), boxes.copy(), confidences.copy()))
                idx = get_best_detection(boxes)
                x1, y1, x2, y2 = map(int, boxes[idx])
                print(f"  ✓ Capture {len(captures)} — box: ({x1},{y1})→({x2},{y2})  conf: {confidences[idx]:.2f}")

        elif key == ord('c'):
            captures.clear()
            print("  Captures cleared.")

        elif key == ord('q'):
            print("Calibration cancelled.")
            cap.release()
            cv2.destroyAllWindows()
            return None

    cap.release()
    cv2.destroyAllWindows()

    if not captures:
        print("No captures recorded — calibration aborted.")
        return None

    # Average pixel height across all captures
    pixel_heights = []
    for _, boxes, _ in captures:
        idx = get_best_detection(boxes)
        x1, y1, x2, y2 = boxes[idx]
        pixel_heights.append(y2 - y1)

    avg_pixel_height = float(np.mean(pixel_heights))
    print(f"\n  Captures used   : {len(captures)}")
    print(f"  Pixel heights   : {[round(h, 1) for h in pixel_heights]}")
    print(f"  Avg pixel height: {avg_pixel_height:.1f}")

    distance = get_float_input("Enter the distance the ball was from the camera (inches): ")

    camera_config = {
        "id": camera_id,
        "camera_fov": int(camera_fov),
        "known_calibration_distance": distance,
        "ball_d_inches": ball_diameter,
        "known_calibration_pixel_height": round(avg_pixel_height, 2),
        "yolo_model_file": yolo_model_path,
        "grayscale": True,
        "margin": 10,
        "min_confidence": 0.5,
    }

    print("\n" + "=" * 60)
    print("CALIBRATION COMPLETE")
    print("=" * 60)
    for k, v in camera_config.items():
        print(f"  {k:<35} = {v}")

    print("\n" + "-" * 60)
    print("Camera object:")
    print("-" * 60)
    print(f"""
camera = Camera(
    id={camera_config['id']},
    camera_fov={camera_config['camera_fov']},
    known_calibration_distance={camera_config['known_calibration_distance']},
    ball_d_inches={camera_config['ball_d_inches']},
    known_calibration_pixel_height={camera_config['known_calibration_pixel_height']},
    yolo_model_file='{camera_config['yolo_model_file']}',
    grayscale={camera_config['grayscale']},
    margin={camera_config['margin']},
    min_confidence={camera_config['min_confidence']}
)""")

    if input("\nSave configuration to file? (y/n): ").strip().lower() == 'y':
        path = Path("camera_config.json")
        path.write_text(json.dumps(camera_config, indent=2))
        print(f"  Saved to {path.resolve()}")

    return camera_config


if __name__ == "__main__":
    yolo_model = input("YOLO model path (default: YoloModels/v26/nano/color-3.1-v26.onnx): ").strip() \
                 or "YoloModels/v26/nano/color-3.1-v26.onnx"

    ball_diam_input = input("Ball diameter in inches (default: 5.90551): ").strip()
    ball_diam = float(ball_diam_input) if ball_diam_input else 5.90551

    calibrate_camera(1, yolo_model, ball_diam)