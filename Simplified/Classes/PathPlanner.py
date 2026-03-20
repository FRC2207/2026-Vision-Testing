import numpy as np
from .CustomDBScan import CustomDBScan

class PathPlanner:
    def __init__(self, fuel_positions, elipson: int, min_samples: int, debug_mode: bool=False):
        # DBSCAN stuff
        self.elipson = elipson
        self.min_samples = min_samples

        self.debug_mode = debug_mode

        # Call at end so everything is initialized before handling outliers and obstacles
        self.fuel_positions, self.noise_positions = self.dbscan(fuel_positions)
    
    def get_noise_positions(self):
        return self.noise_positions
    
    def update_fuel_positions(self, fuel_positions):
        self.fuel_positions, self.noise_positions = self.dbscan(fuel_positions)

        return self.noise_positions, self.fuel_positions
    
    def get_fuel_positions(self):
        return self.fuel_positions

    def dbscan(self, points: list) -> list:
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
            # All points are outliers, but no valid points after filtering
            self.noise_positions = np.array(points[outlier_mask])
            return cleaned_points, self.noise_positions
        else:
            # Some valid points found
            self.noise_positions = np.array(points[outlier_mask])
            return cleaned_points, self.noise_positions