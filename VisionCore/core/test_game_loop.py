import time
import random
import logging
import math
import numpy as np
import pymunk
from VisionCore.utilities.NetworkTableHandler import NetworkTableHandler
from VisionCore.trackers.Fuel import Fuel

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    filemode="w",
    filename="log.txt",
)

# ── Field ──────────────────────────────────────────────────────────────────────
FIELD_X = 16.54
FIELD_Y = 8.21

# ── Balls ──────────────────────────────────────────────────────────────────────
NUM_BALLS        = 300
BALL_RADIUS      = 0.09
BALL_MASS        = 0.15
BALL_ELASTICITY  = 0.5
BALL_FRICTION    = 0.4

# ── Robots ─────────────────────────────────────────────────────────────────────
ROBOT_SIZE       = 0.75
ROBOT_MASS       = 60.0
ROBOT_ELASTICITY = 0.2
ROBOT_FRICTION   = 0.9
MY_ROBOT_SPEED   = 4.5
ENEMY_SPEED      = 3.2
COLLECT_RADIUS   = 0.55
NUM_ENEMIES      = 5

# ── Camera simulation ──────────────────────────────────────────────────────────
CAM_FOV_DEG         = 70
CAM_MAX_DEPTH_M     = 5.0
CAM_MAX_PER_ROW     = 3
CAM_NOISE_M         = 0.02    # gaussian position error, grows with distance
CAM_FALSE_NEG_RATE  = 0.10    # chance a visible ball gets dropped
CAM_FALSE_POS_RATE  = 0.03    # chance each phantom slot fires
CAM_FALSE_POS_COUNT = 4       # max phantoms per frame
CAM_FLICKER_RATE    = 0.05    # chance a detection has a wildly wrong position

NT_IP = "127.0.0.1"

network_handler = NetworkTableHandler(NT_IP)

space = pymunk.Space()
space.damping = 0.25
space.gravity  = (0, 0)

CT_BALL  = 1
CT_ROBOT = 2
CT_WALL  = 3

def add_walls():
    static = space.static_body
    segs = [
        ((0, 0),      (FIELD_X, 0)),
        ((0, 0),      (0, FIELD_Y)),
        ((FIELD_X, 0),(FIELD_X, FIELD_Y)),
        ((0, FIELD_Y),(FIELD_X, FIELD_Y)),
    ]
    for a, b in segs:
        s = pymunk.Segment(static, a, b, 0.05)
        s.elasticity = 0.6
        s.friction   = 0.3
        s.collision_type = CT_WALL
        space.add(s)


def make_ball(x, y):
    body  = pymunk.Body(BALL_MASS, pymunk.moment_for_circle(BALL_MASS, 0, BALL_RADIUS))
    body.position = (x, y)
    shape = pymunk.Circle(body, BALL_RADIUS)
    shape.elasticity     = BALL_ELASTICITY
    shape.friction       = BALL_FRICTION
    shape.collision_type = CT_BALL
    space.add(body, shape)
    return body


def make_robot(x, y):
    inertia = pymunk.moment_for_box(ROBOT_MASS, (ROBOT_SIZE, ROBOT_SIZE))
    body    = pymunk.Body(ROBOT_MASS, inertia)
    body.position = (x, y)
    shape   = pymunk.Poly.create_box(body, (ROBOT_SIZE, ROBOT_SIZE))
    shape.elasticity     = ROBOT_ELASTICITY
    shape.friction       = ROBOT_FRICTION
    shape.collision_type = CT_ROBOT
    space.add(body, shape)
    return body


def random_field_pos(margin=0.8):
    return (
        random.uniform(margin, FIELD_X - margin),
        random.uniform(margin, FIELD_Y - margin),
    )


def scatter_balls():
    balls = []
    for _ in range(NUM_BALLS):
        cx, cy = FIELD_X / 2, FIELD_Y / 2
        x = np.clip(np.random.normal(cx, FIELD_X * 0.3), 0.5, FIELD_X - 0.5)
        y = np.clip(np.random.normal(cy, FIELD_Y * 0.3), 0.5, FIELD_Y - 0.5)
        balls.append(make_ball(float(x), float(y)))
    return balls


def steer_robot(robot_body, target_pos, speed):
    dx = target_pos[0] - robot_body.position.x
    dy = target_pos[1] - robot_body.position.y
    dist = math.hypot(dx, dy)
    if dist < 0.01:
        robot_body.velocity = (0, 0)
        return float(robot_body.angle)
    robot_body.velocity = (dx / dist * speed, dy / dist * speed)
    robot_body.angle    = math.atan2(dy, dx)
    return robot_body.angle


def nearest_unclaimed(robot_body, ball_bodies, claimed):
    pos = np.array(robot_body.position)
    best_dist, best = float("inf"), None
    for b in ball_bodies:
        if id(b) in claimed:
            continue
        d = np.linalg.norm(np.array(b.position) - pos)
        if d < best_dist:
            best_dist, best = d, b
    return best


def simulate_camera(ball_bodies, robot_x, robot_y, robot_yaw):
    half_fov   = math.radians(CAM_FOV_DEG / 2.0)
    bucket_rad = 2 * math.atan2(BALL_RADIUS, CAM_MAX_DEPTH_M)

    sorted_balls = sorted(
        ball_bodies,
        key=lambda b: math.hypot(b.position.x - robot_x, b.position.y - robot_y),
    )

    bucket_counts = {}
    visible = []

    for ball in sorted_balls:
        dx   = ball.position.x - robot_x
        dy   = ball.position.y - robot_y
        dist = math.hypot(dx, dy)

        if dist < 0.1 or dist > CAM_MAX_DEPTH_M:
            continue

        angle_rel = math.atan2(dy, dx) - robot_yaw
        angle_rel = (angle_rel + math.pi) % (2 * math.pi) - math.pi

        if abs(angle_rel) > half_fov:
            continue

        bucket = int(angle_rel / bucket_rad)
        count  = bucket_counts.get(bucket, 0)
        if count >= CAM_MAX_PER_ROW:
            continue
        bucket_counts[bucket] = count + 1

        if random.random() < CAM_FALSE_NEG_RATE:
            continue

        if random.random() < CAM_FLICKER_RATE:
            fx = ball.position.x + random.uniform(-1.5, 1.5)
            fy = ball.position.y + random.uniform(-1.5, 1.5)
        else:
            noise = CAM_NOISE_M * (1.0 + dist / CAM_MAX_DEPTH_M)
            fx = ball.position.x + random.gauss(0, noise)
            fy = ball.position.y + random.gauss(0, noise)

        fx = max(0.1, min(FIELD_X - 0.1, fx))
        fy = max(0.1, min(FIELD_Y - 0.1, fy))
        visible.append(Fuel(float(fx), float(fy)))

    for _ in range(CAM_FALSE_POS_COUNT):
        if random.random() < CAM_FALSE_POS_RATE:
            ph_dist  = random.uniform(0.5, CAM_MAX_DEPTH_M)
            ph_angle = robot_yaw + random.uniform(-half_fov, half_fov)
            ph_x = max(0.1, min(FIELD_X - 0.1, robot_x + ph_dist * math.cos(ph_angle)))
            ph_y = max(0.1, min(FIELD_Y - 0.1, robot_y + ph_dist * math.sin(ph_angle)))
            visible.append(Fuel(float(ph_x), float(ph_y)))

    return visible


if __name__ == "__main__":
    add_walls()
    ball_bodies = scatter_balls()

    try:
        nt_pose = network_handler.get_robot_pose()
        start_x, start_y = nt_pose.X(), nt_pose.Y()
        if start_x == 0.0 and start_y == 0.0:
            raise ValueError("zero pose")
    except Exception:
        start_x, start_y = 2.0, FIELD_Y / 2

    my_robot       = make_robot(start_x, start_y)
    my_robot.angle = 0.0

    enemies = []
    for _ in range(NUM_ENEMIES):
        ex, ey = random_field_pos()
        while math.hypot(ex - start_x, ey - start_y) < 2.5:
            ex, ey = random_field_pos()
        enemies.append(make_robot(ex, ey))

    my_target    = None
    enemy_target = [None] * NUM_ENEMIES

    last_time = time.time()
    tick      = 0

    print("Sim running — Ctrl-C to stop.")

    while True:
        now       = time.time()
        dt        = min(now - last_time, 0.05)
        last_time = now
        tick     += 1

        # Try to pull real pose from NT
        try:
            nt_pose  = network_handler.get_robot_pose()
            real_x   = nt_pose.X()
            real_y   = nt_pose.Y()
            real_yaw = nt_pose.rotation().radians()
            got_nt   = real_x != 0.0 or real_y != 0.0
        except Exception:
            got_nt = False

        if got_nt:
            my_robot.position = (real_x, real_y)
            my_robot.angle    = real_yaw
            cam_x, cam_y, cam_yaw = real_x, real_y, real_yaw
        else:
            cam_x   = float(my_robot.position.x)
            cam_y   = float(my_robot.position.y)
            cam_yaw = float(my_robot.angle)

        # My robot AI (only drives if not being overridden by NT)
        claimed = {id(t) for t in enemy_target if t is not None and t in ball_bodies}
        if my_target not in ball_bodies:
            my_target = None
        if my_target is None:
            my_target = nearest_unclaimed(my_robot, ball_bodies, claimed)
        if my_target is not None:
            claimed.add(id(my_target))
            if not got_nt:
                heading = steer_robot(my_robot, my_target.position, MY_ROBOT_SPEED)
                cam_yaw = heading

        # Enemy AI
        for i, enemy in enumerate(enemies):
            if enemy_target[i] not in ball_bodies:
                enemy_target[i] = None
            if enemy_target[i] is None:
                enemy_target[i] = nearest_unclaimed(enemy, ball_bodies, claimed)
            if enemy_target[i] is not None:
                claimed.add(id(enemy_target[i]))
                steer_robot(enemy, enemy_target[i].position, ENEMY_SPEED)

        space.step(dt)

        # Collect / respawn balls
        all_robots = [my_robot] + enemies
        for ball in ball_bodies:
            for robot in all_robots:
                if math.hypot(
                    ball.position.x - robot.position.x,
                    ball.position.y - robot.position.y,
                ) < COLLECT_RADIUS:
                    ball.position = random_field_pos()
                    ball.velocity = (0, 0)
                    if ball is my_target:
                        my_target = None
                    for i in range(NUM_ENEMIES):
                        if ball is enemy_target[i]:
                            enemy_target[i] = None
                    break

        fuel_list = simulate_camera(ball_bodies, cam_x, cam_y, cam_yaw)
        network_handler.send_fuel_list(fuel_list, "vision_data", "VisionData")

        if tick % 10 == 0:
            print(
                f"\rBalls: {len(ball_bodies):3d} | "
                f"Visible: {len(fuel_list):3d} | "
                f"Pos: ({cam_x:.2f}, {cam_y:.2f}) "
                f"Yaw: {math.degrees(cam_yaw):5.1f}deg | "
                f"NT: {'live' if got_nt else 'sim '}    ",
                end="",
            )

        time.sleep(0.02)