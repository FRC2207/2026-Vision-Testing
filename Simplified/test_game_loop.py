import time
import random
import logging
import numpy as np
import pymunk
from Classes.NetworkTableHandler import NetworkTableHandler
from Classes.Fuel import Fuel

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
NUM_BALLS = 100
NOISE_STDDEV = 0.00  # metres of gaussian (fancy name bruh) noise added to simulate camera error

network_handler = NetworkTableHandler("127.0.0.1")

space = pymunk.Space()
space.damping = 0.3

def add_walls():
    static = space.static_body
    walls = [
        pymunk.Segment(static, (0, 0), (FIELD_X, 0), 0.1),
        pymunk.Segment(static, (0, 0), (0, FIELD_Y), 0.1),
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


def random_field_pos():
    return random.uniform(0.5, FIELD_X - 0.5), random.uniform(0.5, FIELD_Y - 0.5)

if __name__ == "__main__":
    add_walls()

    balls = [make_ball(*random_field_pos()) for _ in range(NUM_BALLS)]
    robots = [
        make_robot(1.0, 1.0),
        make_robot(FIELD_X - 1.0, 1.0),
        make_robot(1.0, FIELD_Y - 1.0),
        make_robot(FIELD_X - 1.0, FIELD_Y - 1.0),
    ]

    last_time = time.time()

    while True:
        now = time.time()
        dt = min(now - last_time, 0.05)
        last_time = now

        # Move each robot toward its nearest ball
        if balls:
            positions = np.array([b.position for b in balls])
            for robot in robots:
                dists = np.linalg.norm(positions - np.array(robot.position), axis=1)
                nearest = positions[np.argmin(dists)]
                delta = nearest - np.array(robot.position)
                dist = np.linalg.norm(delta)
                robot.velocity = (
                    tuple((delta / dist) * ROBOT_SPEED) if dist > 0.01 else (0, 0)
                )

        # Collect balls that are within radius — respawn them at a random position
        for ball in balls:
            for robot in robots:
                if (
                    np.linalg.norm(np.array(ball.position) - np.array(robot.position))
                    < COLLECT_RADIUS
                ):
                    ball.position = random_field_pos()
                    ball.velocity = (0, 0)
                    break

        space.step(dt)

        # Build noisy fuel list and send straight to NT
        fuel_list = [
            Fuel(
                float(b.position.x) + random.gauss(0, NOISE_STDDEV),
                float(b.position.y) + random.gauss(0, NOISE_STDDEV),
            )
            for b in balls
        ]

        network_handler.send_fuel_list(fuel_list, "vision_data", "VisionData")

        time.sleep(0.02)
