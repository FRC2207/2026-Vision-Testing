import cv2
import json
from ultralytics import YOLO
from pathlib import Path

def calibrate_camera(camera_id=0, yolo_model_path="YoloModels/gray-3.0.pt", ball_diameter=9.43):    
    # Get camera FOV
    while True:
        try:
            camera_fov = float(input("\nEnter camera FOV in degrees (e.g., 60): "))
            if 0 < camera_fov < 180:
                break
            print("Please enter a value between 0 and 180 degrees.")
        except ValueError:
            print("Invalid input. Please enter a number.")
    
    # Start camera feed
    cap = cv2.VideoCapture(camera_id)
    if not cap.isOpened():
        print(f"Error: Cannot open camera {camera_id}")
        return None
    
    print("CAMERA FEED ACTIVE")
    print("Position the ball at a known distance from the camera.")
    print("Press SPACE to capture the calibration image.")
    print("Press Q to quit.")
    
    frame = None
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error reading frame")
            break
        
        cv2.imshow("Camera Calibration - Press SPACE to capture, Q to quit", frame)
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord(' '):  # SPACE key
            print("\n✓ Frame captured!")
            break
        elif key == ord('q'):  # Q key
            print("Calibration cancelled.")
            cap.release()
            cv2.destroyAllWindows()
            return None
    
    cap.release()
    cv2.destroyAllWindows()
    
    # Get distance from user
    while True:
        try:
            distance = float(input("Enter the distance you held the ball from the camera (inches): "))
            if distance > 0:
                break
            print("Please enter a positive number.")
        except ValueError:
            print("Invalid input. Please enter a number.")
    
    print("\nRunning YOLO detection...")
    
    # Load YOLO model and detect ball
    model = YOLO(yolo_model_path)
    results = model(frame)[0]
    
    img_h, img_w = results.orig_shape[:2]
    
    if len(results.boxes) == 0:
        print("Error: No ball detected in the image!")
        return None
    
    # Get the largest detection (most likely the ball)
    boxes = results.boxes.xyxy.cpu().numpy()
    confidences = results.boxes.conf.cpu().numpy()
    
    largest_idx = 0
    largest_area = 0
    for i, box in enumerate(boxes):
        x1, y1, x2, y2 = box
        area = (x2 - x1) * (y2 - y1)
        if area > largest_area:
            largest_area = area
            largest_idx = i
    
    # Get bounding box of largest detection
    x1, y1, x2, y2 = boxes[largest_idx]
    conf = confidences[largest_idx]
    
    # Calculate pixel dimensions
    w_pixels = x2 - x1
    h_pixels = y2 - y1
    pixel_height = int(h_pixels)
    
    print(f"\n✓ Ball detected!")
    print(f"  - Confidence: {conf:.2f}")
    print(f"  - Bounding box: ({int(x1)}, {int(y1)}) to ({int(x2)}, {int(y2)})")
    print(f"  - Pixel height: {pixel_height}")
    print(f"  - Pixel width: {int(w_pixels)}")
    
    # Calculate calibration constants
    known_calibration_distance = distance
    known_calibration_pixel_height = pixel_height
    
    # Create camera configuration
    camera_config = {
        "id": camera_id,
        "camera_fov": int(camera_fov),
        "known_calibration_distance": known_calibration_distance,
        "ball_d_inches": ball_diameter,
        "known_calibration_pixel_height": known_calibration_pixel_height,
        "yolo_model_file": yolo_model_path,
        "grayscale": True,
        "margin": 10,
        "min_confidence": 0.5
    }
    
    # Print results
    print("\n" + "=" * 70)
    print("CALIBRATION COMPLETE - CAMERA OBJECT PARAMETERS")
    print("=" * 70)
    
    for key, value in camera_config.items():
        print(f"  {key:35} = {value}")
    
    print("\n" + "-" * 70)
    print("To create a Camera object, use:")
    print("-" * 70)
    print(f"""
from Classes.Camera import Camera

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
)
""")
    
    # Save to file
    save = input("\nSave configuration to file? (y/n): ").lower()
    if save == 'y':
        filename = "camera_config.json"
        with open(filename, 'w') as f:
            json.dump(camera_config, f, indent=2)
        print(f"✓ Configuration saved to {filename}")
    
    return camera_config


if __name__ == "__main__":
    # Get user inputs
    camera_id = 0
    
    yolo_model = input("Enter YOLO model path (default: YoloModels/gray-3.0.pt): ").strip()
    if not yolo_model:
        yolo_model = "YoloModels/gray-3.0.pt"
    
    ball_diam = input("Enter ball diameter in inches (default: 5.90551): ").strip()
    if not ball_diam:
        ball_diam = 5.90551
    else:
        ball_diam = float(ball_diam)
    
    # Run calibration
    config = calibrate_camera(camera_id, yolo_model, ball_diam)