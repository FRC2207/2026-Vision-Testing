class AprilTagCamera:
    def __init__(self, config):
        self.config = config
        self.logger = logging.getLogger("VisionCore.AprilTagCamera")