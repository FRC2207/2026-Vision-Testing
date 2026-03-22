from .Fuel import Fuel
from .CustomDBScan import CustomDBScan
import numpy as np

class FuelTracker:
    def __init__(self, fuel_list: list[Fuel] = None, distance_threshold: float = 0.127):
        self.fuel_list = fuel_list or []
        self.distance_threshold = distance_threshold

    def update(self, new_fuel_list: list[Fuel]) -> list[Fuel]:
        self.fuel_list = self._deduplicate(new_fuel_list)
        return self.fuel_list
    def _deduplicate(self, fuels: list[Fuel]) -> list[Fuel]:
        if not fuels:
            return []

        positions = np.array([f.get_position() for f in fuels])

        scanner = CustomDBScan(positions, eps=self.distance_threshold, samples=1)
        labels = scanner.get_dbscan()

        # Merge each cluster into its centroid
        merged = {}
        for label, pos in zip(labels, positions):
            merged.setdefault(label, []).append(pos)

        return [
            Fuel(*np.mean(g, axis=0).tolist())
            for g in merged.values()
        ]

    def get_fuel_list(self) -> list[Fuel]:
        return self.fuel_list