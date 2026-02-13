# Game loop file that should be run on the pi for multiple cameras
from Classes.Camera import Camera
from Classes.PathPlanner import PathPlanner
from Classes.NetworkTableHandler import NetworkTableHandler
import time
import constants
from Classes.FuelTracker import FuelTracker

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

network_handler = NetworkTableHandler("10.22.7.2") # Pretty sure this is right


if __name__ == "__main__":
    print("[Custom Fuel Intake] Creating simplified loop.")

    # Create planner
    starting_positions = camera1.run()
    fuel_tracker = FuelTracker(starting_positions, constants.DISTANCE_THRESHOLD)

    planner = PathPlanner(
        starting_positions, constants.STARTING_POSITION,
        constants.ELIPSON, constants.MIN_SAMPLES,
        debug_mode=constants.DEBUG_MODE,
    )

    while True:
        # This update to run multiple cameras if probaly gonna destroy fps
        fuel_positions1 = camera1.run()
        fuel_positions2 = camera2.run()
        fuel_positions3 = camera3.run()
        fuel_positions4 = camera4.run()

        fuel_tracker.set_fuel_list(fuel_positions1)
        fuel_positions = fuel_tracker.update(fuel_positions2)
        fuel_positions = fuel_tracker.update(fuel_positions3)
        fuel_positions = fuel_tracker.update(fuel_positions4)

        start_time = time.perf_counter()
        vision_end_time = time.perf_counter()

        if len(fuel_positions) == 0:
            print("[Custom Fuel Intake] No fuel positions detected. Skipping loop iteration.")
            continue
        else:
            print(f"[Custom Fuel Intake] Detected fuels: {len(fuel_positions)}")

        _, fuel_positions = planner.update_fuel_positions(fuel_positions)
        network_handler.send_data(fuel_positions, "fuel_data", "yolo_data")

        end_time = time.perf_counter()
        print(f"[Custom Fuel Intake] Vision time: {vision_end_time - start_time}. Loop time: {end_time - start_time}")

    # camera.destroy()