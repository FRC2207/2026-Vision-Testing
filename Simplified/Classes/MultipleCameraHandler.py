from Classes.Camera import Camera
import cv2
import numpy as np
import logging
import concurrent.futures
import threading

class MultipleCameraHandler:
    def __init__(self, cameras: list[Camera], vision_model_path: str):
        self.cameras = cameras
        self.logger = logging.getLogger(__name__)
        self._annotated_frames: list = [None] * len(cameras)
        self._latest_positions: list = [np.empty((0, 2))] * len(cameras)
        self._locks = [threading.Lock() for _ in cameras]
        self._stopped = False

        # Each camera gets its own persistent worker thread
        for i, cam in enumerate(cameras):
            threading.Thread(
                target=self._camera_loop,
                args=(i, cam),
                daemon=True
            ).start()

    def _camera_loop(self, i: int, camera: Camera):
        while not self._stopped:
            try:
                positions, annotated = camera.run()
                with self._locks[i]:
                    self._latest_positions[i] = positions if positions is not None else np.empty((0, 2))
                    self._annotated_frames[i] = annotated
            except Exception as e:
                self.logger.warning(f"Camera {camera.source} error: {e}")

    def predict(self) -> np.ndarray:
        # Non-blocking — just reads latest cached results
        all_positions = []
        for i in range(len(self.cameras)):
            with self._locks[i]:
                pos = self._latest_positions[i]
            if pos is not None and len(pos) > 0:
                all_positions.append(pos)

        return np.vstack(all_positions) if all_positions else np.empty((0, 2))

    def get_combined_frame(self):
        frames = []
        for i, cam in enumerate(self.cameras):
            with self._locks[i]:
                f = self._annotated_frames[i]
            if f is None:
                f = cam.get_frame()
            if f is not None:
                frames.append(f)

        if not frames:
            return None
        if len(frames) == 1:
            return frames[0]

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
        self._stopped = True
        for cam in self.cameras:
            cam.destroy()