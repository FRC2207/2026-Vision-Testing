from Classes.Camera import Camera
import numpy as np
from GenericYolo.genericYolo import Box, Results, YoloWrapper

class MultipleCameraHandler:
    """This will only work if all cameras run the same vision model"""
    def __init__(self, cameras: list[Camera], vision_model_path: str):
        self.cameras = cameras
        self.frames = []

        self.vision_model_path = vision_model_path
        self.model = YoloWrapper(self.vision_model_path)

    def predict(self):
        frames = [cam.get_frame() for cam in self.cameras]
        
        results = self.model.predict(frames)
        feild_positions = []
        for i, result in enumerate(results):
            feild_positions.extend(self.cameras[i].run_with_supplied_data(result))

        return feild_positions
    
    def get_combined_frame(self):
        frames = [cam.get_frame() for cam in self.cameras]

        combined_frame = np.hstack(frames)
        return combined_frame
    
    def destroy(self):
        for cam in self.cameras:
            cam.destroy()