# Game loop file that should be run on the pi for multiple cameras
from Classes.Camera import Camera
from Classes.PathPlanner import PathPlanner
from Classes.NetworkTableHandler import NetworkTableHandler
import time
import constants
from Classes.FuelTracker import FuelTracker
from Classes.Fuel import Fuel
import logging
import numpy as np
from Classes.CameraApp import CameraApp
import threading

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
    subsystem="field"
)

camera2 = Camera(
    1,
    constants.CAMERA_FOV, constants.KNOWN_CALIBRATION_DISTANCE,
        constants.BALL_D_INCHES, constants.KNOWN_CALIBRATION_PIXEL_HEIGHT,
    constants.YOLO_MODEL_FILE, 10, 1, grayscale=constants.GRAYSCALE,
    debug_mode=constants.DEBUG_MODE,
    subsystem="field"
)

camera3 = Camera(
    2,
    constants.CAMERA_FOV, constants.KNOWN_CALIBRATION_DISTANCE,
        constants.BALL_D_INCHES, constants.KNOWN_CALIBRATION_PIXEL_HEIGHT,
    constants.YOLO_MODEL_FILE, 10, 1, grayscale=constants.GRAYSCALE,
    debug_mode=constants.DEBUG_MODE,
    subsystem="hopper"
)

camera4 = Camera(
    3,
    constants.CAMERA_FOV, constants.KNOWN_CALIBRATION_DISTANCE,
        constants.BALL_D_INCHES, constants.KNOWN_CALIBRATION_PIXEL_HEIGHT,
    constants.YOLO_MODEL_FILE, 10, 1, grayscale=constants.GRAYSCALE,
    debug_mode=constants.DEBUG_MODE,
    subsystem="outtake"
)

network_handler = NetworkTableHandler(constants.NETWORKTABLES_IP)

if constants.APP_MODE:
    camera_app = CameraApp()
    threading.Thread(target=camera_app.run, daemon=True).start()

def numpy_to_fuel_list(fuel_positions):
    return [Fuel(p[0], p[1]) for p in fuel_positions]

def fuel_list_to_numpy(fuel_list: list[Fuel]):
    return np.array([fuel.get_position() for fuel in fuel_list])

if __name__ == "__main__":
    try:
        logger.info("Creating simplified loop.")

        # Create planner
        fuel_tracker = FuelTracker([], constants.DISTANCE_THRESHOLD)    
        planner = PathPlanner(
            [], constants.STARTING_POSITION,
            constants.ELIPSON, constants.MIN_SAMPLES,
            debug_mode=constants.DEBUG_MODE,
        )

        while True:
            # This update to run multiple cameras its probaly gonna destroy fps
            outtake_positions = []
            hopper_positions = []

            fuel_tracker.set_fuel_list([])
            for camera in [camera1, camera2, camera3, camera4]:
                if camera.subsystem == "hopper":
                    hopper_positions = numpy_to_fuel_list(camera.run())
                elif camera.subsystem == "outtake":
                    outtake_positions = numpy_to_fuel_list(camera.run())
                else:
                    fuel_tracker.update(numpy_to_fuel_list(camera.run()))

            network_handler.send_data(True if len(hopper_positions) > 0 else False, "hopper_sees_object", "yolo_data")

            # start_time = time.perf_counter()
            # vision_end_time = time.perf_counter()

            fuel_positions = fuel_list_to_numpy(fuel_tracker.fuel_list) # Cause im dumb and didnt make pathplanner work with Fuel objects
            if len(fuel_positions) == 0:
                logger.info(f"No fuel positions detected. Skipping loop iteration.")
                continue
            else:
                logger.info(f"Detected fuels: {len(fuel_positions)}")

            if constants.APP_MODE:
                _, frame = camera.get_yolo_data()
                camera_app.set_frame(frame)

            noise_positions, fuel_positions = planner.update_fuel_positions(fuel_positions)
            fuel_objects = numpy_to_fuel_list(fuel_positions)
            network_handler.send_data(fuel_objects, "field_positions", "yolo_data")

            # end_time = time.perf_counter()
            # logger.info(f"Vision time: {vision_end_time - start_time}. Loop time: {end_time - start_time}")
    finally:
        camera1.destroy()
        camera2.destroy()
        camera3.destroy()
        camera4.destroy()