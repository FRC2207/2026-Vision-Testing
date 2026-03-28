import numpy as np
from .Fuel import Fuel
import VisionCoreConfig

class FuelTracker:
    def __init__(self, config: VisionCoreConfig):
        self.fuel_list = []
        self.distance_threshold = config["distance_threshold"]

    def update(self, new_fuel_list: list[Fuel]):
        for fuel in self.fuel_list:
            fuel.update()
        
        self.fuel_list = [f for f in self.fuel_list if not f.destroyed]

        self.add_fuel_list(new_fuel_list)
        return self.fuel_list

    def add_fuel_list(self, fuels: list[Fuel]):
        for fuel in fuels:
            if not self._fuel_already_exists(fuel, self.fuel_list):
                self.fuel_list.append(fuel)

    def _fuel_already_exists(self, new_fuel: Fuel, existing_fuels: list[Fuel]) -> bool:
        if not existing_fuels:
            return False
        
        new_pos = np.array(new_fuel.get_position())
        for existing in existing_fuels:
            existing_pos = np.array(existing.get_position())
            if np.linalg.norm(new_pos - existing_pos) < self.distance_threshold:
                existing.reset_time()
                return True
        return False

    def get_fuel_list(self) -> list[Fuel]:
        return self.fuel_list