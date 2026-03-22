from Classes.Camera import Camera
import cv2
import numpy as np
import logging
import concurrent.futures
import threading

class MultipleCameraHandler:
    def __init__(self, cameras, vision_model_path):
        self.cameras = cameras
        self.logger = logging.getLogger(__name__)
        self._annotated_frames = [None] * len(cameras)
        self._latest_positions = [np.empty((0, 2))] * len(cameras)
        self._locks = [threading.Lock() for _ in cameras]
        self._new_data_event = threading.Event()  # ← fires when ANY camera finishes
        self._stopped = False

        for i, cam in enumerate(cameras):
            threading.Thread(target=self._camera_loop, args=(i, cam), daemon=True).start()

    def _camera_loop(self, i, camera):
        while not self._stopped:
            try:
                positions, annotated = camera.run()
                with self._locks[i]:
                    self._latest_positions[i] = positions if positions is not None else np.empty((0, 2))
                    self._annotated_frames[i] = annotated
                self._new_data_event.set()  # ← wake up predict()
            except Exception as e:
                self.logger.warning(f"Camera {camera.source} error: {e}")

    def predict(self) -> np.ndarray:
        self._new_data_event.wait(timeout=0.1)  # ← blocks until a camera finishes
        self._new_data_event.clear()

        all_positions = []
        for i in range(len(self.cameras)):
            with self._locks[i]:
                pos = self._latest_positions[i]
            if pos is not None and len(pos) > 0:
                all_positions.append(pos)

        return np.vstack(all_positions) if all_positions else np.empty((0, 2))