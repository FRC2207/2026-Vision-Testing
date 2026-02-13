import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import splprep, splev
import json
from matplotlib.animation import FuncAnimation
from matplotlib.patches import Rectangle, Polygon
from sklearn.cluster import DBSCAN

# Robot constants
ROBOT_WIDTH = 15
ROBOT_HEIGHT = 20
INTAKE_WIDTH = 17
STARTING_POSITION = np.array([0, 100])

# Fuel constants
BALL_DIAMETER = 6

# Field Constants
MIN_X = -50
MAX_X = 50
MIN_Y = 0
MAX_Y = 100

# Other
ANIMATE = True
COLLECTION_PRIORITY = 0.35  # 0.0 = smoothest, 1.0 = hit all balls exactly
SMOOTHING_FACTOR = 35  # Lower = closer to waypoints, Higher = smoother
DEGREE = 3

# Spline Stuff/DBSCAN Stuff
ELIPSON = 10
MIN_SAMPLES = 5

PRESET = 0
if PRESET == 1:
    # Waypoint
    COLLECTION_PRIORITY = 1
    SMOOTHING_FACTOR = 0
    DEGREE = 1
elif PRESET == 2:
    # Smoothest
    COLLECTION_PRIORITY = 0
    SMOOTHING_FACTOR = 50
    DEGREE = 5
elif PRESET == 3:
    # Optimal
    COLLECTION_PRIORITY = 0.5
    SMOOTHING_FACTOR = 35
    DEGREE = 3

def load_data(filename: str):
    with open(filename, 'r') as f:
        data_json = json.load(f)
    points = data_json.get("points", [])
    return np.array([[p['x'], p['y']] for p in points])

FUEL_POSITIONS = load_data("ball_layout.json")

class CustomDBScan:
    def __init__(self, points: list, eps: int, samples: int):
        self.points = points
        self.eps = eps
        self.samples = samples
        self.dbscan = DBSCAN(eps=self.eps, min_samples=self.samples)
        
    def get_dbscan(self):
        clusters = self.dbscan.fit_predict(self.points)
        return clusters
    
class Obstacle:
    def __init__(self, x: int, y: int, width: int, height: int):
        self.x = x
        self.y = y
        self.width = width
        self.height = height

    def is_point_in_obstacle(self, pos_x: int, pos_y: int):
        # self.x and self.y is the top left corner
        return (self.x <= pos_x <= self.x + self.width) and (self.y <= pos_y <= self.y + self.height)

    def line_intersects_obstacle(self, line_start, line_end, rect_top_left, rect_dim):
        x1, y1 = line_start
        x2, y2 = line_end
        rx, ry = rect_top_left
        rw, rh = rect_dim
        
        # Rectangle boundaries
        left, top = rx, ry
        right, bottom = rx + rw, ry + rh

        # 1. Check if either end point is inside the rectangle
        def is_inside(px, py):
            return left <= px <= right and top <= py <= bottom

        if is_inside(x1, y1) or is_inside(x2, y2):
            return True

        # 2. Check intersection with the 4 sides of the rectangle
        # Sides: (x3, y3) to (x4, y4)
        sides = [
            ((left, top), (right, top)),    # Top
            ((right, top), (right, bottom)), # Right
            ((right, bottom), (left, bottom)), # Bottom
            ((left, bottom), (left, top))  # Left
        ]

        for p3, p4 in sides:
            if self.line_segments_intersect(line_start, line_end, p3, p4):
                return True

        return False

    def line_segments_intersect(self, p1, p2, p3, p4):
        x1, y1 = p1
        x2, y2 = p2
        x3, y3 = p3
        x4, y4 = p4

        denom = (y4-y3)*(x2-x1) - (x4-x3)*(y2-y1)
        if denom == 0: return False # Parallel
        
        ua = ((x4-x3)*(y1-y3) - (y4-y3)*(x1-x3)) / denom
        ub = ((x2-x1)*(y1-y3) - (y2-y1)*(x1-x3)) / denom

        # Intersection occurs if 0 <= ua <= 1 and 0 <= ub <= 1
        return 0 <= ua <= 1 and 0 <= ub <= 1

OBSTACLES = [
    Obstacle(10, 50, 10, 50),
    Obstacle(-30, 30, 10, 5)
]

class PathPlanner:
    def __init__(self, fuel_positions, start_pos, robot_width, intake_width, obstacles: list[Obstacle] = []):
        self.fuel_positions = fuel_positions
        self.current_pos = start_pos
        self.robot_width = robot_width
        self.intake_width = intake_width
        self.visited = set()
        self.obstacles = obstacles
        self.path = [self.current_pos.copy()]

    def path_with_rotation(path):
        poses = []
        for i in range(len(path)):
            if i < len(path) - 1:
                d = path[i+1] - path[i]
            else:
                d = path[i] - path[i-1]

            theta = np.degrees(np.arctan2(d[1], d[0]))
            poses.append([path[i][0], path[i][1], theta])
        return np.array(poses)

    def nearest_neighbor_path(self):
        remaining = list(range(len(self.fuel_positions)))
        
        while remaining:
            distances = [np.linalg.norm(self.fuel_positions[i] - self.current_pos) 
                        for i in remaining]
            
            nearest_idx = remaining[np.argmin(distances)]
            remaining.remove(nearest_idx)
            
            self.current_pos = self.fuel_positions[nearest_idx].copy()
            self.path.append(self.current_pos.copy())
            self.visited.add(nearest_idx)

        # for i, point in enumerate(self.path):
        #     if not (len(self.path) >= i + 1):
        #         break
        #     for obstacle in self.obstacles:
        #         line_startpoint, line_endpoint = point[i], point[i + 1]
        #         if obstacle.line_intersects_obstacle(line_startpoint, line_endpoint):
        #             # Somehow implement a* search to get to the other side of the obstacle and get to the point
        #             pass
        
        return np.array(self.path)
        
    def smooth_path(self, smoothing_factor=0.01):
        path = np.array(self.path)

        unique_path = [path[0]]
        for i in range(1, len(path)):
            if not np.allclose(path[i], path[i-1]):
                unique_path.append(path[i])
        path = np.array(unique_path)

        if len(path) < 4:
            return path, None, None

        adjusted_s = len(path) * smoothing_factor * (1 - COLLECTION_PRIORITY)

        tck, u = splprep([path[:, 0], path[:, 1]], s=adjusted_s, k=DEGREE)
        u_smooth = np.linspace(0, 1, len(path) * 20)
        smooth_x, smooth_y = splev(u_smooth, tck)

        result_path = np.column_stack([smooth_x, smooth_y])
        result_path[0] = path[0]

        return result_path, tck, u_smooth
    
    def calculate_path_length(self, path):
        distances = np.sqrt(np.sum(np.diff(path, axis=0)**2, axis=1))
        return np.sum(distances)
    
    def path_with_rotation_from_spline(self, tck, u_vals, smooth_path):
        dx, dy = splev(u_vals, tck, der=1)
        thetas = np.degrees(np.arctan2(dy, dx))

        poses = []
        for i in range(len(smooth_path)):
            poses.append([smooth_path[i][0], smooth_path[i][1], thetas[i]])
        return np.array(poses)
    
def handle_outlier_points_and_obstacles(points: list) -> list:
    safe_point = np.ones(len(points), dtype=bool)
    for i, point in enumerate(points):
        for obstacle in OBSTACLES:
            if obstacle.is_point_in_obstacle(point[0], point[1]):
                    safe_point[i] = False

    points = points[safe_point]
    
    dbscan = CustomDBScan(points, eps=ELIPSON, samples=MIN_SAMPLES)
    labels = dbscan.get_dbscan()
    
    non_noise_mask = (labels != -1)
    outlier_mask = ~non_noise_mask
    
    cleaned_points = points[non_noise_mask]
    if (len(cleaned_points.tolist()) == 0):
        # No outliers
        return points, np.empty((0, 2))
    else:
        NOISE_POSITIONS = np.array(points[outlier_mask])

        return cleaned_points, NOISE_POSITIONS

def plot_path_planning(fuel_positions, noise_positions, raw_path, smooth_path, start_pos, obstacles: list[Obstacle]=[], animation_speed=1.0):
    fig, ax = plt.subplots(figsize=(14, 10))

    # Setting up the graph
    ax.set_xlim(MIN_X, MAX_X)
    ax.set_ylim(MIN_Y, MAX_Y)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.2)
    ax.set_title('2026 FRC Fuel Collection', fontsize=14, fontweight='bold')
    ax.set_xlabel('X Position (units)')
    ax.set_ylabel('Y Position (units)')
    ax.add_patch(plt.Rectangle((MIN_X, MIN_Y), MAX_X-MIN_X, MAX_Y-MIN_Y,
                              fill=False, edgecolor='black', linewidth=2))
    
    # Plot the smooth path
    ax.plot(smooth_path[:, 0], smooth_path[:, 1], 'purple', linewidth=3, 
           label='Robot Path', zorder=3)
    
    # Plot the raw direct path
    ax.plot(raw_path[:, 0], raw_path[:, 1], 'orange', linestyle='--', linewidth=2, 
           label='Raw Waypoints', zorder=2)
    
    # Fuel
    fuel_colors = ['red'] * len(fuel_positions)
    ax.scatter(fuel_positions[:, 0], fuel_positions[:, 1], 
              s=400*(BALL_DIAMETER)/36, c=fuel_colors, marker='o', label='Fuel', zorder=5, 
              edgecolor='darkred', linewidth=2.5)
    
    noise_colors = ['black'] * len(noise_positions)
    ax.scatter(noise_positions[:, 0], noise_positions[:, 1], 
              s=400*(BALL_DIAMETER)/36, c=noise_colors, marker='o', label='Noise', zorder=5, 
              edgecolor='gray', linewidth=2.5)

    # Start position marker
    ax.scatter(*start_pos, s=250, c='green', marker='s', label='Start Position', zorder=5,
              edgecolor='darkgreen', linewidth=2.5)

    ax.legend(loc='upper right', fontsize=11, framealpha=0.95)
    
    num_frames = len(smooth_path) if ANIMATE else 1
    interval = max(1, int(20 / animation_speed)) if ANIMATE else 1

    for obstacle in obstacles:
        rect = Rectangle((obstacle.x, obstacle.y), obstacle.width, obstacle.height, 
                         color='gray', alpha=0.7, label='Obstacle', zorder=4)
        ax.add_patch(rect)

    if ANIMATE:
        # Arrow to represent the robot
        arrow = ax.quiver(
            smooth_path[0,0], smooth_path[0,1],
            np.cos(np.radians(pose_path[0,2])),
            np.sin(np.radians(pose_path[0,2])),
            scale=20, width=0.008, color="blue"
        )

        num_frames = len(smooth_path)
        interval = max(1, int(20 / animation_speed))
    
    def animate(frame):
        x, y, theta = pose_path[frame]

        dx = np.cos(np.radians(theta))
        dy = np.sin(np.radians(theta))

        arrow.set_offsets([x, y])
        arrow.set_UVC(dx, dy)

        return arrow,
    
    if ANIMATE:
        _ = FuncAnimation(fig, animate, frames=num_frames, interval=interval, blit=True, repeat=True)
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    FUEL_POSITIONS, NOISE_POSITIONS = handle_outlier_points_and_obstacles(FUEL_POSITIONS)

    print("=" * 60)
    print("2026 FRC Path Planner Thingie")
    print("=" * 60)

    # Create planner
    planner = PathPlanner(
        FUEL_POSITIONS, 
        STARTING_POSITION, 
        ROBOT_WIDTH,
        INTAKE_WIDTH,
        obstacles=OBSTACLES
    )
    
    print(f"\nField created with {len(FUEL_POSITIONS)} fuel positions")
    print(f"Starting position: {STARTING_POSITION}")
    
    raw_path = planner.nearest_neighbor_path()
    print(f"\nPath generated visiting all {len(FUEL_POSITIONS)} fuel positions")
    print(f"  Raw path distance: {planner.calculate_path_length(raw_path):.2f} units")
    
    smooth_path, tck, u_vals = planner.smooth_path(smoothing_factor=SMOOTHING_FACTOR)
    pose_path = planner.path_with_rotation_from_spline(tck, u_vals, smooth_path)
    print(f"Swerve path created")
    print(f"  Smooth path distance: {planner.calculate_path_length(smooth_path):.2f} units")
    print(f"  Path waypoints: {len(smooth_path)}")

    print(f"\nGenerating path visualization{' with animation' if ANIMATE else ''}...")
    plot_path_planning(FUEL_POSITIONS, NOISE_POSITIONS, raw_path, smooth_path, STARTING_POSITION, 
                    obstacles=OBSTACLES, animation_speed=100)