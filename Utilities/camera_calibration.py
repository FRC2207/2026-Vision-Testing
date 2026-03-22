import cv2
import json
import numpy as np
from ultralytics import YOLO
from pathlib import Path

def ask_float(prompt, condition=lambda x: x > 0, error="Invalid, try again."):
    while True:
        try:
            val = float(input(prompt))
            if condition(val):
                return val
            print(error)
        except ValueError:
            print("That's not a number, try again.")

def get_largest_box(boxes):
    areas = [(x2 - x1) * (y2 - y1) for x1, y1, x2, y2 in boxes]
    return int(np.argmax(areas))


def draw_crosshair(frame):
    h, w = frame.shape[:2]
    cx, cy = w // 2, h // 2

    # Big center lines
    cv2.line(frame, (cx, 0), (cx, h), (255, 255, 255), 1)
    cv2.line(frame, (0, cy), (w, cy), (255, 255, 255), 1)

    # Little center box
    box_size = 20
    cv2.rectangle(frame, (cx - box_size, cy - box_size), (cx + box_size, cy + box_size), (255, 255, 255), 1)


def draw_boxes(frame, boxes, confs):
    for box, conf in zip(boxes, confs):
        x1, y1, x2, y2 = map(int, box)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, f"{conf:.2f}", (x1, y1 - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2)


def calibrate(camera_id=0, model_path="YoloModels/gray-3.0.pt", ball_diameter=9.43):
    fov = ask_float(
        "\nWhat's the camera FOV in degrees? (e.g. 60): ",
        condition=lambda x: 0 < x < 180,
        error="Has to be between 0 and 180."
    )

    print("\nLoading model...")
    model = YOLO(model_path)

    cap = cv2.VideoCapture(camera_id)
    if not cap.isOpened():
        print(f"Couldn't open camera {camera_id}")
        return None

    print("\nCamera is live! Line the ball up with the crosshair.")
    print("  SPACE = capture  |  C = clear captures  |  Q = done capturing\n")

    captures = []

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Lost the camera feed")
            break

        results = model(frame, verbose=False)[0]
        boxes = results.boxes.xyxy.cpu().numpy()
        confs = results.boxes.conf.cpu().numpy()

        draw_crosshair(frame)
        draw_boxes(frame, boxes, confs)

        # little hud in the corner
        cv2.putText(frame, f"detections: {len(boxes)}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 200, 255), 2)
        cv2.putText(frame, f"captures: {len(captures)}", (10, 58), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        cv2.putText(frame, "SPACE: snap  C: clear  Q: finish", (10, 86), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1)

        cv2.imshow("Calibration", frame)
        key = cv2.waitKey(1) & 0xFF

        if key == ord(' '):
            if len(boxes) == 0:
                print("  no ball detected, try again")
            else:
                captures.append((frame.copy(), boxes.copy(), confs.copy()))
                idx = get_largest_box(boxes)
                x1, y1, x2, y2 = map(int, boxes[idx])
                print(f"  capture {len(captures)} — ({x1},{y1})→({x2},{y2})  conf: {confs[idx]:.2f}")

        elif key == ord('c'):
            captures.clear()
            print("  cleared all captures")

        elif key == ord('q'):
            if len(captures) == 0:
                print("No captures, quitting.")
                cap.release()
                cv2.destroyAllWindows()
                return None
            break

    cap.release()
    cv2.destroyAllWindows()

    # average out the pixel heights from all captures
    heights = []
    for _, boxes, _ in captures:
        idx = get_largest_box(boxes)
        x1, y1, x2, y2 = boxes[idx]
        heights.append(y2 - y1)

    avg_height = float(np.mean(heights))

    print(f"\n  captures : {len(captures)}")
    print(f"  heights  : {[round(h, 1) for h in heights]}")
    print(f"  average  : {avg_height:.1f}px")

    distance = ask_float("How far was the ball from the camera? (inches): ")

    config = {
        "id": camera_id,
        "camera_fov": int(fov),
        "known_calibration_distance": distance,
        "ball_d_inches": ball_diameter,
        "known_calibration_pixel_height": round(avg_height, 2),
        "yolo_model_file": model_path,
        "grayscale": True,
        "margin": 10,
        "min_confidence": 0.5,
    }

    print("\n" + "=" * 55)
    print("DONE! Here's your camera config:")
    print("=" * 55)
    for k, v in config.items():
        print(f"  {k:<35} = {v}")

    print("\n" + "-" * 55)
    print("Paste this into your code:")
    print("-" * 55)
    print(f"""
camera = Camera(
    id={config['id']},
    camera_fov={config['camera_fov']},
    known_calibration_distance={config['known_calibration_distance']},
    ball_d_inches={config['ball_d_inches']},
    known_calibration_pixel_height={config['known_calibration_pixel_height']},
    yolo_model_file='{config['yolo_model_file']}',
    grayscale={config['grayscale']},
    margin={config['margin']},
    min_confidence={config['min_confidence']}
)""")

    if input("\nSave to camera_config.json? (y/n): ").strip().lower() == 'y':
        path = Path("camera_config.json")
        path.write_text(json.dumps(config, indent=2))
        print(f"  saved to {path.resolve()}")

    return config


if __name__ == "__main__":
    model = input("Model path (default: YoloModels/v26/nano/color-3.1-v26.onnx): ").strip() \
            or "YoloModels/v26/nano/color-3.1-v26.onnx"

    diam = input("Ball diameter in inches (default: 5.90551): ").strip()
    diam = float(diam) if diam else 5.90551

    calibrate(1, model, diam)