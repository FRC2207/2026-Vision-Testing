from Classes.PathPlanner import PathPlanner
from Classes.NetworkTableHandler import NetworkTableHandler
import time
import constants
from Classes.Fuel import Fuel
from Classes.FuelTracker import FuelTracker
import random
import numpy as np
import json
import logging
import pymunk

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    filemode="w",
    filename="log.txt",
)

FIELD_X = 16.54
FIELD_Y = 8.21
ROBOT_SPEED = 2.0
COLLECT_RADIUS = 0.4

network_handler = NetworkTableHandler("127.0.0.1")

space = pymunk.Space()
space.damping = 0.3

def generate_fuel(n=300):
    return [Fuel(random.uniform(0.5, FIELD_X - 0.5), random.uniform(0.5, FIELD_Y - 0.5)) for _ in range(n)]

def add_walls():
    static = space.static_body
    walls = [
        pymunk.Segment(static, (0, 0),       (FIELD_X, 0),       0.1),
        pymunk.Segment(static, (0, 0),       (0, FIELD_Y),       0.1),
        pymunk.Segment(static, (FIELD_X, 0), (FIELD_X, FIELD_Y), 0.1),
        pymunk.Segment(static, (0, FIELD_Y), (FIELD_X, FIELD_Y), 0.1),
    ]
    for w in walls:
        w.elasticity = 0.6
        w.friction = 0.5
    space.add(*walls)

def make_ball(x, y):
    body = pymunk.Body(mass=1, moment=pymunk.moment_for_circle(1, 0, 0.1))
    body.position = x, y
    shape = pymunk.Circle(body, radius=0.1)
    shape.elasticity = 0.6
    shape.friction = 0.5
    space.add(body, shape)
    return body

def make_robot(x, y):
    body = pymunk.Body(mass=20, moment=pymunk.moment_for_box(20, (0.7, 0.7)))
    body.position = x, y
    shape = pymunk.Poly.create_box(body, size=(0.7, 0.7))
    shape.elasticity = 0.3
    shape.friction = 0.8
    space.add(body, shape)
    return body

def get_fuel():
    with open("ball_layout.json") as file:
        data = json.load(file)
    return [Fuel(p["x"], p["y"]) for p in data["points"]]

def numpy_to_fuel_list(positions):
    return [Fuel(p[0], p[1]) for p in positions]

def fuel_list_to_numpy(fuel_list):
    return np.array([f.get_position() for f in fuel_list])

if __name__ == "__main__":
    add_walls()

    try:
        # initial_fuel = get_fuel()
        initial_fuel = generate_fuel()
    except Exception as e:
        logging.error(f"Failed to load ball_layout.json: {e}")
        initial_fuel = []

    balls = [make_ball(f.get_position()[0], f.get_position()[1]) for f in initial_fuel]
    original_layout = [(f.get_position()[0], f.get_position()[1]) for f in initial_fuel]

    # robots = [
    #     make_robot(1.0,           1.0),
    #     make_robot(FIELD_X - 1.0, 1.0),
    #     make_robot(1.0,           FIELD_Y - 1.0),
    #     make_robot(FIELD_X - 1.0, FIELD_Y - 1.0),
    # ]
    robots = []

    raw_fuel_positions = fuel_list_to_numpy(initial_fuel)
    planner = PathPlanner(
        raw_fuel_positions,
        constants.ELIPSON,
        constants.MIN_SAMPLES,
        constants.DEBUG_MODE
    )
    fuel_tracker = FuelTracker(initial_fuel, constants.DISTANCE_THRESHOLD)

    last_time = time.time()

    while True:
        now = time.time()
        dt = min(now - last_time, 0.05)
        last_time = now

        for robot in robots:
            if not balls:
                break
            positions = np.array([b.position for b in balls])
            dists = np.linalg.norm(positions - np.array(robot.position), axis=1)
            nearest = positions[np.argmin(dists)]
            direction = nearest - np.array(robot.position)
            dist = np.linalg.norm(direction)
            if dist > 0.01:
                robot.velocity = tuple((direction / dist) * ROBOT_SPEED)
            else:
                robot.velocity = (0, 0)

        to_remove = []
        for ball in balls:
            for robot in robots:
                d = np.linalg.norm(
                    np.array(ball.position) - np.array(robot.position)
                )
                if d < COLLECT_RADIUS:
                    to_remove.append(ball)
                    break

        for ball in to_remove:
            if ball in balls:
                # respawn at random position instead of removing
                ball.position = (
                    random.uniform(0.5, FIELD_X - 0.5),
                    random.uniform(0.5, FIELD_Y - 0.5)
                )
                ball.velocity = (0, 0)

        if len(balls) < 5:
            logging.info("Field cleared — respawning balls")
            for x, y in original_layout:
                balls.append(make_ball(x, y))

        space.step(dt)
        fuel_positions_fuel_list = [Fuel(float(b.position.x), float(b.position.y)) for b in balls]
        # fuel_positions_fuel_list = [
        #     Fuel(
        #         float(b.position.x) + random.gauss(0, 0.03),
        #         float(b.position.y) + random.gauss(0, 0.03)
        #     )
        #     for b in balls
        # ]

        fuel_tracker.set_fuel_list(fuel_positions_fuel_list)
        fuel_tracker.sort()
        fuel_positions_fuel_list = fuel_tracker.get_fuel_list()

        raw_fuel_positions = fuel_list_to_numpy(fuel_positions_fuel_list)
        if raw_fuel_positions.size > 0:
            _, updated = planner.update_fuel_positions(raw_fuel_positions)
            fuel_positions_fuel_list = numpy_to_fuel_list(updated)

        network_handler.send_fuel_list(
            fuel_positions_fuel_list, "vision_data", "VisionData"
        )

        time.sleep(0.02)