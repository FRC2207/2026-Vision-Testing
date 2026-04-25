# This is a example usage of how to use VisionCore
# Made by the one and only Aidan Jensen (yes he is real I know I seem fake cause I mog so hard but I promis I'm real)
from VisionCore import VisionCore
from VisionCore.vision.ObjectDetectionCamera import ObjectDetectionCamera
from VisionCore.config.VisionCoreConfig import VisionCoreConfig
from VisionCore.validations.ez import unit_tests

def main():
    config = VisionCoreConfig("config.json")

    cameras = [
        ObjectDetectionCamera(
            config.camera_config("Microsoft Cinema"),
            config
        )
    ]

    vision = VisionCore(cameras, config)

    vision.run()

if __name__ == "__main__":
    if not unit_tests():
        raise SystemExit("Unit tests failed")
    main()