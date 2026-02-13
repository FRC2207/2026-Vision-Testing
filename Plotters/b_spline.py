import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import splprep, splev
import json
from matplotlib.animation import FuncAnimation

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
COLLECTION_PRIORITY = 0.75  # 0.0 = smoothest, 1.0 = hit all balls exactly
SMOOTHING_FACTOR = 50  # Lower = closer to waypoints, Higher = smoother

# Spline Stuff
DEGREE = 3

PRESET = 3
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

class PathPlanner:
    def __init__(self, fuel_positions, start_pos, robot_width, intake_width):
        self.fuel_positions = fuel_positions
        self.current_pos = start_pos
        self.robot_width = robot_width
        self.intake_width = intake_width
        self.visited = set()
        self.path = [self.current_pos.copy()]
        
    def point_to_segment_distance(p, a, b):
        ap = p - a
        ab = b - a
        t = np.dot(ap, ab) / np.dot(ab, ab)
        t = np.clip(t, 0, 1)
        closest = a + t * ab
        return np.linalg.norm(p - closest)

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

        return np.column_stack([smooth_x, smooth_y]), tck, u_smooth
    
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

def plot_path_planning(fuel_positions, raw_path, smooth_path, start_pos, intake_width=17, animation_speed=1.0):
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
    fuel_scatter = ax.scatter(fuel_positions[:, 0], fuel_positions[:, 1], 
              s=400*(BALL_DIAMETER)/36, c=fuel_colors, marker='o', label='Fuel', zorder=5, 
              edgecolor='darkred', linewidth=2.5)
    
    # Start position marker
    ax.scatter(*start_pos, s=250, c='green', marker='s', label='Start Position', zorder=5,
              edgecolor='darkgreen', linewidth=2.5)
    
    ax.legend(loc='upper right', fontsize=11, framealpha=0.95)
    
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
    print("=" * 60)
    print("2026 FRC Path Planner Thingie")
    print("=" * 60)
    
    # Create planner
    planner = PathPlanner(
        FUEL_POSITIONS, 
        STARTING_POSITION, 
        ROBOT_WIDTH,
        INTAKE_WIDTH
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
    plot_path_planning(FUEL_POSITIONS, raw_path, smooth_path, STARTING_POSITION, 
                      intake_width=INTAKE_WIDTH, animation_speed=100)