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
            for fuel in self.fuel_list:
                if np.linalg.norm(new_fuel.get_position(), fuel) <= self.distance_threshold:
                    new_fuel.set_id(fuel.get_id())
                else:
                    new_fuel.set_id(self.get_highest_id())

        self.fuel_list = new_fuel_list

    def set_fuel_list(self, fuels: list[Fuel]):
        self.fuel_list = fuels