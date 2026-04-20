import logging
from VisionCore import VisionCore

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

print("Welcome to VisionCore! Boot will setup vision automatically, and run for 30 seconds. Boot in progress...")
logger.info("Welcome to VisionCore! Boot will setup vision automatically, and run for 30 seconds. Boot in progress...")

vision = VisionCore()
vision.run(duration=30)