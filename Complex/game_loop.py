# Game loop file that should be run on the pi
from Classes.Camera import Camera
from Classes.PathPlanner import PathPlanner
import numpy as np
import time
import constants

# Camera class
camera = Camera(
    "http://photon1.local:1184/stream.mjpg",
    constants.CAMERA_FOV, constants.KNOWN_CALIBRATION_DISTANCE,
        constants.BALL_D_INCHES, constants.KNOWN_CALIBRATION_PIXEL_HEIGHT,
    constants.YOLO_MODEL_FILE, 90, 1, grayscale=constants.GRAYSCALE,
    debug_mode=constants.DEBUG_MODE,
)

if __name__ == "__main__":
    print("[Custom Fuel Intake] Creating loop. Getting initial fuel positions and creating custom PathPlanner.")

    # Create planner
    fuel_positions = camera.run()
    planner = PathPlanner(
        fuel_positions, constants.STARTING_POSITION,
        constants.ELIPSON, constants.MIN_SAMPLES, constants.DEGREE, constants.COLLECTION_PRIORITY, constants.SMOOTHING_FACTOR,
        obstacles=constants.OBSTACLES,
        debug_mode=constants.DEBUG_MODE,
        auto_adjust=constants.AUTO_ADJUST
    )

    while True:
        start_time = time.perf_counter()
        fuel_positions = camera.run()
        vision_end_time = time.perf_counter()

        if len(fuel_positions) == 0:
            print("[Custom Fuel Intake] No fuel positions detected. Skipping loop iteration.")
            continue
        else:
            print(f"[Custom Fuel Intake] Detected fuels: {len(fuel_positions)}")

        noise_positions, raw_path, _ = planner.update_fuel_positions(fuel_positions)

        end_time = time.perf_counter()
        print(f"[Custom Fuel Intake] Vision time: {vision_end_time - start_time}. Loop time: {end_time - start_time}")

    camera.destroy()