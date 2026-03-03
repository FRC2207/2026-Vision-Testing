import cv2
from ultralytics import YOLO
import time


def run_detection(video_source: int, model_file_path: str):
    playing = True
    model = YOLO(model_file_path, task="detect", verbose=False)

    cap = cv2.VideoCapture(video_source)
    if not cap.isOpened():
        print(f"Error: Could not open video source {video_source}")
        return

    prev_frame_time = 0
    new_frame_time = 0

    while True:
        if playing:
            success, frame = cap.read()
            if not success:
                break

            results = model.predict(
                source=frame,
                stream=True,
                imgsz=320,
                vid_stride=3,
                agnostic_nms=True,
                verbose=False
            )
            for r in results:
                img = r.orig_img
                for box in r.boxes.xyxy:
                    x1, y1, x2, y2 = map(int, box)
                    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            # annotated_frame = results[0].plot()

            new_frame_time = time.time()
            fps = 1 / (new_frame_time - prev_frame_time)
            prev_frame_time = new_frame_time
            fps_text = f"FPS: {fps:.2f}"

            cv2.putText(frame, fps_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

            cv2.namedWindow('Random Yolo Vision Thingie', cv2.WINDOW_NORMAL)

            cv2.resizeWindow('Random Yolo Vision Thingie', 600, 600)
            cv2.imshow("Random Yolo Vision Thingie", frame)

        key = cv2.waitKey(25) & 0xFF
        if key == ord('q'): break
        if key == ord(' '): playing = not playing

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    run_detection("Images/PXL_20260228_155842030.mp4", fr"YoloModels\v26\nano\clean\color-3.1-v26_openvino_model")