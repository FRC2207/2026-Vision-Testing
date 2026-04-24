from VisionCore import VisionCore
from VisionCore.vision.Camera import Camera
from VisionCore.config import VisionCoreConfig

config = VisionCoreConfig("config.json")

cameras = [
    Camera(config.camera_config("Microsoft Cinema"), config)
]

vision = VisionCore(cameras, config)
vision.run()