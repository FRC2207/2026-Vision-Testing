# This is a example usage of how to use VisionCore
# Made by the one and only Aidan Jensen (yes he is real I know I seem fake cause I mog so hard but I promis I'm real)
from VisionCore import VisionCore
from VisionCore.vision.ObjectDetectionCamera import ObjectDetectionCamera
from VisionCore.config.VisionCoreConfig import VisionCoreConfig
import VisionCore.core.constants as constants

def main():
    # Load config (you can also customize this)
    config = VisionCoreConfig(constants.CONFIG)

    # Create cameras (add/remove depending on your setup)
    cameras = [
        ObjectDetectionCamera(
            constants.CONFIG.camera_config("Camera 1"),
            constants.CONFIG,
        ),
        # Uncomment for multi-camera
        # Camera(
        #     constants.CONFIG.camera_config("Camera 2"),
        #     constants.CONFIG,
        # ),
    ]

    vision = VisionCore(cameras, config)

    vision.run()


if __name__ == "__main__":
    main()