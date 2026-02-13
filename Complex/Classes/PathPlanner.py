import numpy as np
from .Obstacle import Obstacle
from scipy.interpolate import splprep, splev
import threading
import matplotlib as plt
from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation
from .CustomDBScan import CustomDBScan

class PathPlanner:
    def __init__(self, fuel_positions, start_pos, elipson: int, min_samples: int, degree: int, collection_priority: int, smoothing_factor: int, obstacles: list[Obstacle] = [], debug_mode: bool=False, auto_adjust: bool=True):
        self.starting_pos = start_pos
        self.visited = set()
        self.obstacles = obstacles
        self.path = [self.starting_pos]

        # Graphing stuff
        self.elipson = elipson
        self.min_samples = min_samples
        self.degree = degree
        self.collection_priority = collection_priority
        self.smoothing_factor = smoothing_factor

        self.debug_mode = debug_mode
        self.auto_adjust = auto_adjust

        self.starting_degree = self.degree

        # Call at end so everything is initialized before handling outliers and obstacles
        self.fuel_positions, self.noise_positions = self.handle_outlier_points_and_obstacles(fuel_positions)
        
        if self.debug_mode:
            self.start_ploting_frame_thread()

    def plot_frame(self):
        fig, ax = plt.subplots(figsize=(10, 7))

        # Basic axes setup I'll add the dynamic stuff later maybe
        ax.set_xlim(-50, 50)
        ax.set_ylim(0, 100)
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.2)
        ax.set_title('Real-time ploty thingie', fontsize=14, fontweight='bold')
        ax.set_xlabel('X Position (units)')
        ax.set_ylabel('Y Position (units)')

        # Initial empty plotting things it should update every frame
        path_line, = ax.plot([], [], color='purple', linewidth=3, label='Robot Path', zorder=3)
        fuel_scatter = ax.scatter([], [], s=120, c='red', edgecolor='darkred', linewidth=1.5, label='Fuel', zorder=5)
        noise_scatter = ax.scatter([], [], s=80, c='black', edgecolor='gray', linewidth=1.0, label='Noise', zorder=5)
        start_scatter = ax.scatter([], [], s=200, c='green', marker='s', edgecolor='darkgreen', linewidth=1.5, label='Start Position', zorder=6)

        ax.legend(loc='upper right', fontsize=10, framealpha=0.95)

        def animate(_frame):
            # Convert path to numpy array
            path = np.array(self.path) if len(self.path) > 0 else np.empty((0, 2))

            # Update path line
            if path.size:
                path_line.set_data(path[:, 0], path[:, 1])
            else:
                path_line.set_data([], [])

            # Update fuel and noise positions from the instance (safe empty fallback)
            if getattr(self, 'fuel_positions', None) is not None and len(self.fuel_positions):
                fuel_scatter.set_offsets(np.asarray(self.fuel_positions))
            else:
                fuel_scatter.set_offsets(np.empty((0, 2)))

            if getattr(self, 'noise_positions', None) is not None and len(self.noise_positions):
                noise_scatter.set_offsets(np.asarray(self.noise_positions))
            else:
                noise_scatter.set_offsets(np.empty((0, 2)))

            start_scatter.set_offsets([self.starting_pos])
            # Return artists that have changed for blitting
            return path_line, fuel_scatter, noise_scatter, start_scatter

        # Animation interval in milliseconds; tune as needed
        interval_ms = getattr(self, 'plot_interval_ms', 200)

        # Create animation (runs until the window is closed)
        _anim = FuncAnimation(fig, animate, interval=interval_ms, blit=True)

        plt.tight_layout()
        plt.show()

    def start_ploting_frame_thread(self):
        t = threading.Thread(target=self.plot_frame)

        t.start()

    def path_with_rotation(self, path):
        poses = []
        for i in range(len(path)):
            if i < len(path) - 1:
                d = path[i+1] - path[i]
            else:
                d = path[i] - path[i-1]

            theta = np.degrees(np.arctan2(d[1], d[0]))
            poses.append([path[i][0], path[i][1], theta])
        return np.array(poses)
    
    def get_noise_positions(self):
        return self.noise_positions

    def nearest_neighbor_path(self):
        remaining = list(range(len(self.fuel_positions)))

        current_pos = self.starting_pos
        
        while remaining:
            distances = [np.linalg.norm(self.fuel_positions[i] - current_pos) 
                        for i in remaining]
            
            nearest_idx = remaining[np.argmin(distances)]
            remaining.remove(nearest_idx)
            
            current_pos = self.fuel_positions[nearest_idx].copy()
            self.path.append(current_pos.copy())
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
        
    def smooth_path(self):
        path = np.array(self.path)

        unique_path = [path[0]]
        for i in range(1, len(path)):
            if not np.allclose(path[i], path[i-1]):
                unique_path.append(path[i])
        path = np.array(unique_path)

        if len(path) <= self.degree:
            adjusted_degree = max(1, len(path) - 1)
            if self.debug_mode:
                print(f"[Custom Path Planner] Not enough unique points after dedup ({len(path)} points). Reducing degree from {self.degree} to {adjusted_degree}")
        else:
            adjusted_degree = self.degree

        if len(path) <= adjusted_degree:
            if self.debug_mode:
                print(f"[Custom Path Planner] Cannot apply spline with {len(path)} points and degree {adjusted_degree}. Returning path without smoothing.")
            # Return path as-is without spline interpolation, so a crappy waypoint path
            return path, None, None

        adjusted_s = len(path) * self.smoothing_factor * (1 - self.collection_priority)

        tck, u = splprep([path[:, 0], path[:, 1]], s=adjusted_s, k=adjusted_degree)
        u_smooth = np.linspace(0, 1, len(path) * 20)
        smooth_x, smooth_y = splev(u_smooth, tck)

        result_path = np.column_stack([smooth_x, smooth_y])
        result_path[0] = path[0]

        return result_path, tck, u_smooth
    
    def calculate_path_length(self, path):
        distances = np.sqrt(np.sum(np.diff(path, axis=0)**2, axis=1))
        return np.sum(distances)
    
    def update_fuel_positions(self, fuel_positions):
        self.fuel_positions, self.noise_positions = self.handle_outlier_points_and_obstacles(fuel_positions)

        # Reset degree
        self.degree = self.starting_degree

        # Degree auto adjusting stuff
        if len(fuel_positions) + 1 < self.degree:
            if self.auto_adjust:
                self.degree = len(fuel_positions) - 1
                self.degree = max(1, self.degree)
                if self.debug_mode:
                    print(f"[Custom Path Planner] Because auto adjust was true, degree was set to {self.degree}")
                    if self.degree == 1:
                        print(f"[Custom Path Planner] Degree is set to 1 because {len(fuel_positions)} were detected. Path won't generated good.")
            else:
                return self.path, None, None

        self.path = [self.starting_pos]
        
        raw_path = self.nearest_neighbor_path()
        
        if (len(fuel_positions) > self.degree):
            smooth_path, tck, u_vals = self.smooth_path()
            # If tck is None, it means it coudln't apply spline, use non-spline rotation
            if tck is None:
                pose_path = self.path_with_rotation(smooth_path)
            else:
                pose_path = self.path_with_rotation_from_spline(tck, u_vals, smooth_path)
        else: # Auto adjust screwed me over
            smooth_path, tck, u_vals = self.path, None, None
            # pose_path = self.path_with_rotation(smooth_path)
            pose_path = raw_path

        self.path = pose_path

        return self.noise_positions, raw_path, pose_path

    def path_with_rotation_from_spline(self, tck, u_vals, smooth_path):
        if smooth_path is None or tck is None or u_vals is None:
            return np.array([[0, 0, 0]])
        dx, dy = splev(u_vals, tck, der=1)
        thetas = np.degrees(np.arctan2(dy, dx))

        poses = []
        for i in range(len(smooth_path)):
            poses.append([smooth_path[i][0], smooth_path[i][1], thetas[i]])
        return np.array(poses)
    
    def get_path_for_network_tabels(self):
        nn_path = self.nearest_neighbor_path()

        

    def handle_outlier_points_and_obstacles(self, points: list) -> list:
        safe_point = np.ones(len(points), dtype=bool)
        for obstacle in self.obstacles:
            for i, point in enumerate(points):
                if obstacle.is_point_in_obstacle(point[0], point[1]):
                    safe_point[i] = False

        points = points[safe_point]

        if len(points) == 0:
            # No valid points
            self.noise_positions = np.empty((0, 2))
            return points, self.noise_positions
        
        dbscan = CustomDBScan(points, eps=self.elipson, samples=self.min_samples)
        labels = dbscan.get_dbscan()
        
        non_noise_mask = (labels != -1)
        outlier_mask = ~non_noise_mask
        
        cleaned_points = points[non_noise_mask]
        if len(cleaned_points) == 0:
            # All points are outliers, but no valid points after obstacle filtering
            self.noise_positions = np.array(points[outlier_mask])
            return cleaned_points, self.noise_positions
        else:
            # Some valid points found
            self.noise_positions = np.array(points[outlier_mask])
            return cleaned_points, self.noise_positions