from .Fuel import Fuel
import numpy as np

class FuelTracker:
    def __init__(self, fuel_list: list[Fuel] = None, distance_threshold: int = 5):
        self.fuel_list = fuel_list or []
        self.distance_threshold = distance_threshold  # Min distance inches to treat two detections as distinct fuels

    def update(self, new_fuel_list: list[Fuel]) -> list[Fuel]:
        self.fuel_list = self._deduplicate(new_fuel_list)
        self._sort()
        return self.fuel_list

    def _deduplicate(self, fuels: list[Fuel]) -> list[Fuel]:
        groups = []
        for fuel in fuels:
            for group in groups:
                if np.linalg.norm(fuel.get_position() - group[0].get_position()) <= self.distance_threshold:
                    group.append(fuel)
                    break
            else:
                groups.append([fuel])

        merged = []
        for group in groups:
            avg_pos = np.mean([f.get_position() for f in group], axis=0)
            group[0].set_position(avg_pos)
            merged.append(group[0])

        return merged

    def _sort(self):
        self.fuel_list.sort(key=lambda fuel: np.linalg.norm(fuel.get_position()))

    def get_fuel_list(self) -> list[Fuel]:
        return self.fuel_list