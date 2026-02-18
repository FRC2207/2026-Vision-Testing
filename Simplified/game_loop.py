# Game loop file that should be run on the pi for multiple cameras
from Classes.Camera import Camera
from Classes.PathPlanner import PathPlanner
from Classes.NetworkTableHandler import NetworkTableHandler
import time
import constants
from Classes.FuelTracker import FuelTracker
from Classes.Fuel import Fuel
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    filemode="w", # Append dont overwrite
    filename="log.txt"
)

logger = logging.getLogger(__name__)

# Camera class
camera1 = Camera(
    0,
    constants.CAMERA_FOV, constants.KNOWN_CALIBRATION_DISTANCE,
        constants.BALL_D_INCHES, constants.KNOWN_CALIBRATION_PIXEL_HEIGHT,
    constants.YOLO_MODEL_FILE, 10, 1, grayscale=constants.GRAYSCALE,
    debug_mode=constants.DEBUG_MODE,
)

camera2 = Camera(
    1,
    constants.CAMERA_FOV, constants.KNOWN_CALIBRATION_DISTANCE,
        constants.BALL_D_INCHES, constants.KNOWN_CALIBRATION_PIXEL_HEIGHT,
    constants.YOLO_MODEL_FILE, 10, 1, grayscale=constants.GRAYSCALE,
    debug_mode=constants.DEBUG_MODE,
)

camera3 = Camera(
    2,
    constants.CAMERA_FOV, constants.KNOWN_CALIBRATION_DISTANCE,
        constants.BALL_D_INCHES, constants.KNOWN_CALIBRATION_PIXEL_HEIGHT,
    constants.YOLO_MODEL_FILE, 10, 1, grayscale=constants.GRAYSCALE,
    debug_mode=constants.DEBUG_MODE,
)

camera4 = Camera(
    3,
    constants.CAMERA_FOV, constants.KNOWN_CALIBRATION_DISTANCE,
        constants.BALL_D_INCHES, constants.KNOWN_CALIBRATION_PIXEL_HEIGHT,
    constants.YOLO_MODEL_FILE, 10, 1, grayscale=constants.GRAYSCALE,
    debug_mode=constants.DEBUG_MODE,
)

network_handler = NetworkTableHandler(constants.NETWORKTABLES_IP)

def numpy_to_fuel_list(fuel_positions):
    return [Fuel(p[0], p[1]) for p in fuel_positions]

if __name__ == "__main__":
    try:
        logger.info("Creating simplified loop.")

        # Create planner
        starting_positions = camera1.run()
        fuel_tracker = FuelTracker([Fuel(p[0], p[1]) for p in starting_positions], constants.DISTANCE_THRESHOLD)    
        planner = PathPlanner(
            starting_positions, constants.STARTING_POSITION,
            constants.ELIPSON, constants.MIN_SAMPLES,
            debug_mode=constants.DEBUG_MODE,
        )

        while True:
            # This update to run multiple cameras its probaly gonna destroy fps
            fuel_positions1 = numpy_to_fuel_list(camera1.run())
            fuel_positions2 = numpy_to_fuel_list(camera2.run())
            fuel_positions3 = numpy_to_fuel_list(camera3.run())
            fuel_positions4 = numpy_to_fuel_list(camera4.run())

            fuel_tracker.set_fuel_list(fuel_positions1)
            fuel_positions = fuel_tracker.update(fuel_positions2)
            fuel_positions = fuel_tracker.update(fuel_positions3)
            fuel_positions = fuel_tracker.update(fuel_positions4)

            start_time = time.perf_counter()
            vision_end_time = time.perf_counter()

            if len(fuel_positions) == 0:
                logger.info(f"No fuel positions detected. Skipping loop iteration.")
                continue
            else:
                logger.info(f"Detected fuels: {len(fuel_positions)}")

            noise_positions, raw_path, smooth_path = planner.update_fuel_positions(fuel_positions)
            network_handler.send_data(smooth_path, "smooth_path", "yolo_data")

            end_time = time.perf_counter()
            logger.info(f"Vision time: {vision_end_time - start_time}. Loop time: {end_time - start_time}")
    finally:
        camera1.destroy()
        camera2.destroy()
        camera3.destroy()
        camera4.destroy()