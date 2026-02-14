# Game loop file that should be run on the pi for one camera
from Classes.Camera import Camera
from Classes.PathPlanner import PathPlanner
from Classes.NetworkTableHandler import NetworkTableHandler
import time
import constants

# Camera class
camera = Camera(
    0,
    constants.CAMERA_FOV, constants.KNOWN_CALIBRATION_DISTANCE,
        constants.BALL_D_INCHES, constants.KNOWN_CALIBRATION_PIXEL_HEIGHT,
    constants.YOLO_MODEL_FILE, 10, 1, grayscale=constants.GRAYSCALE,
    debug_mode=constants.DEBUG_MODE,
)

network_handler = NetworkTableHandler(constants.NETWORKTABLES_IP)

if __name__ == "__main__":
    try:
        print("[Custom Fuel Intake] Creating simplified loop.")

        # Create planner
        try:
            fuel_positions = camera.run()
        except:
            fuel_positions = []
        planner = PathPlanner(
            fuel_positions, constants.STARTING_POSITION,
            constants.ELIPSON, constants.MIN_SAMPLES,
            debug_mode=constants.DEBUG_MODE,
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

            _, fuel_positions = planner.update_fuel_positions(fuel_positions)
            network_handler.send_data(fuel_positions, "fuel_data", "yolo_data")

            end_time = time.perf_counter()
            print(f"[Custom Fuel Intake] Vision time: {vision_end_time - start_time}. Loop time: {end_time - start_time}")
    finally:
        camera.destroy()