import json
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
import numpy as np
from sklearn.metrics.pairwise import euclidean_distances
import matplotlib.cm as cm
import splines
from scipy.interpolate import splprep, splev

def load_data(filename: str):
    with open(filename, 'r') as f:
        data_json = json.load(f)
    points = data_json.get("points", [])
    meta_data = data_json.get("metadata", {"xrange": [-50, 50], "yrange": [0, 100], "count": 0})
    coords = np.array([[p['x'], p['y']] for p in points])
    return coords, meta_data.get("xrange", [-50, 50]), meta_data.get("yrange", [0, 100])

def plot_path_planning(fuel_positions, smooth_path, start_pos):
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.set_xlim(X_MIN, X_MAX)
    ax.set_ylim(Y_MIN, Y_MAX)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    ax.set_title('2026 FRC Fuel Collection Path', fontsize=14, fontweight='bold')

    # Plot the smooth path
    if len(smooth_path) > 0:
        path_np = np.array(smooth_path)
        ax.plot(path_np[:, 0], path_np[:, 1], 'purple', linewidth=3, label='Planned Path', zorder=3)

    # Fuel
    ax.scatter(fuel_positions[:, 0], fuel_positions[:, 1], s=50, c='red', marker='o', label='Fuel', zorder=5, edgecolor='darkred')
    
    ax.scatter(start_pos[0, 0], start_pos[0, 1], s=200, c='green', marker='s', label='Start', zorder=6, edgecolor='black')
    
    ax.legend(loc='upper right')
    plt.show()

class CustomNN:
    def __init__(self, points: list, STARTING_POS):
        self.points = points
        self.STARTING_POS = STARTING_POS
        
    def get_points(self):
        if len(self.points) == 0: return []
        unvisited = [p for p in self.points]
        pos = self.STARTING_POS.copy()[0]
        path = [pos]
        
        while unvisited:
            distances = [np.linalg.norm(p - pos) for p in unvisited]
            nearest_idx = np.argmin(distances)
            pos = unvisited.pop(nearest_idx)
            path.append(pos)
            
        return np.array(path)

    def plot(self):
        pass

class CustomDBScan:
    def __init__(self, points: list, eps: int, samples: int):
        self.points = points
        self.eps = eps
        self.samples = samples
        self.dbscan = DBSCAN(eps=self.eps, min_samples=self.samples)
        
    def get_dbscan(self):
        clusters = self.dbscan.fit_predict(self.points)
        return clusters

class CustomBSpline:
    def __init__(self, points: list, t: float, c: float, b: float):
        self.points = np.array(points)
        self.t = t
        self.c = c
        self.b = b
        
    def get_spl(self):
        if len(self.points) < 2: 
            return self.points, self.points
        
        sorted_idx = np.argsort(self.points[:, 0])
        ordered_points = self.points[sorted_idx]
        
        if len(ordered_points) < 3:
            return ordered_points, ordered_points

        kb_spline = splines.KochanekBartels(ordered_points, tcb=(self.t, self.c, self.b))
        
        t_vals = np.linspace(0, len(ordered_points) - 1, 100)
        path = np.array([kb_spline.evaluate(t) for t in t_vals])
        
        return path, ordered_points

if __name__ == "__main__":
    np.random.seed(42)
    FUEL_POSITIONS = np.random.uniform(-40, 90, (50, 2))
    X_RANGE, Y_RANGE = [-50, 50], [0, 100]
    X_MIN, X_MAX = X_RANGE[0], X_RANGE[1]
    Y_MIN, Y_MAX = Y_RANGE[0], Y_RANGE[1]
    
    STARTING_POS = np.array([[0, 0]])
    EPSILON = 15
    MIN_SAMPLES = 3
    DRIVE_SPEED = 10

    dbscan = CustomDBScan(FUEL_POSITIONS, EPSILON, MIN_SAMPLES)
    clusters = dbscan.get_dbscan()

    unique_clusters = set(clusters)
    grouped_data = {}
    for cluster_id in unique_clusters:
        if cluster_id == -1: continue
        mask = (clusters == cluster_id)
        positions = FUEL_POSITIONS[mask]
        
        distances = euclidean_distances(positions, STARTING_POS)
        avg_dist = np.mean(distances)
        
        grouped_data[int(cluster_id)] = {
            "points": positions,
            "value": len(positions) / (avg_dist / DRIVE_SPEED) if avg_dist > 0 else 0
        }

    sorted_clusters = sorted(grouped_data.items(), key=lambda item: item[1]["value"], reverse=True)

    total_path = []
    current_pos = STARTING_POS.copy()[0]

    for cluster_id, data in sorted_clusters:
        cluster_points = data["points"]
        
        nn = CustomNN(cluster_points, np.array([current_pos]))
        ordered_cluster_points = nn.get_points()
        
        bspline = CustomBSpline(ordered_cluster_points, -0.5, 0.5, 0.5) 
        path, _ = bspline.get_spl()
        
        if len(total_path) == 0:
            total_path.extend(path)
        else:
            total_path.extend(path)
            
        current_pos = path[-1]

    plot_path_planning(FUEL_POSITIONS, total_path, STARTING_POS)
