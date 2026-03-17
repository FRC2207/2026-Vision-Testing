from Classes.Camera import Camera
import cv2
import numpy as np
import logging


class MultipleCameraHandler:
    """Runs multiple cameras that share the same vision model (each Camera owns its own model instance)."""
    def __init__(self, cameras: list[Camera], vision_model_path: str):
        self.cameras = cameras
        self.logger = logging.getLogger(__name__)
        self.vision_model_path = vision_model_path

    def predict(self) -> np.ndarray:
        all_positions = []
        for camera in self.cameras:
            try:
                positions, _ = camera.run()
                if positions is not None and len(positions) > 0:
                    all_positions.append(positions)
            except Exception as e:
                self.logger.warning(f"Camera {camera.source} failed during predict: {e}")

        if all_positions:
            return np.vstack(all_positions)
        return np.empty((0, 2))

    def get_combined_frame(self):
        frames = [cam.get_frame() for cam in self.cameras]
        valid = [f for f in frames if f is not None]
        if not valid:
            return None
        if len(valid) == 1:
            return valid[0]

        target_h = min(f.shape[0] for f in valid)
        resized = []
        for f in valid:
            h, w = f.shape[:2]
            if h != target_h:
                scale = target_h / h
                new_w = int(w * scale)
                f = cv2.resize(f, (new_w, target_h), interpolation=cv2.INTER_AREA)
            resized.append(f)

        return np.hstack(resized)

    def destroy(self):
        for cam in self.cameras:
            cam.destroy()