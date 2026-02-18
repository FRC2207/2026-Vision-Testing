from .Fuel import Fuel
import numpy as np

class FuelTracker:
    def __init__(self, fuel_list: list[Fuel], distance_threshold: int=5):
        self.fuel_list = fuel_list
        self.distance_threshold = distance_threshold # The amount of error in inches allowed for a fuel to keep the id

        self.init()

    def init(self):
        for i, fuel in enumerate(self.fuel_list):
            fuel.set_id(i)

    def get_highest_id(self):
        highest = float("-inf")

        for fuel in self.fuel_list:
            if fuel.get_id() > highest:
                highest = fuel.get_id()

        return highest

    def update(self, new_fuel_list: list[Fuel]):
        for new_fuel in new_fuel_list:
            found_match = False
            for fuel in self.fuel_list:
                distance = np.linalg.norm(new_fuel.get_position() - fuel.get_position())
                if distance <= self.distance_threshold:
                    new_fuel.set_id(fuel.get_id())
                    found_match = True
                    break
            
            if not found_match:
                new_fuel.set_id(self.get_highest_id() + 1)
                self.fuel_list.append(new_fuel)
        
        return self.fuel_list

    def set_fuel_list(self, fuels: list[Fuel]):
        self.fuel_list = fuels