from Classes.Camera import Camera
import cv2
import numpy as np
import logging
import concurrent.futures


class MultipleCameraHandler:
    def __init__(self, cameras: list[Camera], vision_model_path: str):
        self.cameras = cameras
        self.logger = logging.getLogger(__name__)
        self.vision_model_path = vision_model_path
        self._annotated_frames: list = [None] * len(cameras)

        self._executor = concurrent.futures.ThreadPoolExecutor(max_workers=len(cameras))

    def _run_camera(self, args):
        i, camera = args
        positions, annotated = camera.run()
        return i, positions, annotated

    def predict(self) -> np.ndarray:
        all_positions = []
        
        futures = {
            self._executor.submit(self._run_camera, (i, cam)): i
            for i, cam in enumerate(self.cameras)
        }

        for future in concurrent.futures.as_completed(futures, timeout=0.15): # .15s max wait
            try:
                i, positions, annotated = future.result()
                self._annotated_frames[i] = annotated
                if positions is not None and len(positions) > 0:
                    all_positions.append(positions)
            except Exception as e:
                i = futures[future]
                self.logger.warning(f"Camera {self.cameras[i].source} failed during predict: {e}")
                self._annotated_frames[i] = None

        if all_positions:
            return np.vstack(all_positions)
        return np.empty((0, 2))

    def get_combined_frame(self):
        frames = []
        for i, cam in enumerate(self.cameras):
            f = self._annotated_frames[i]
            if f is None:
                f = cam.get_frame()  # fallback to raw
            if f is not None:
                frames.append(f)

        if not frames:
            return None
        if len(frames) == 1:
            return frames[0]

        # Normalize heights before stacking
        target_h = min(f.shape[0] for f in frames)
        resized = []
        for f in frames:
            h, w = f.shape[:2]
            if h != target_h:
                new_w = int(w * (target_h / h))
                f = cv2.resize(f, (new_w, target_h), interpolation=cv2.INTER_AREA)
            resized.append(f)

        return np.hstack(resized)

    def destroy(self):
        self._executor.shutdown(wait=False)
        for cam in self.cameras:
            cam.destroy()