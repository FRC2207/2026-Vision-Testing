from scipy.spatial import spatial

class Fuel:
    def __init__(self):
        pass

    def set_id(self, id: int):
        self.id = id

class FuelTracker:
    def __init__(self, fuel_list: list[Fuel], distance_threshold: int=2):
        self.fuel_list = fuel_list
        self.distance_threshold = distance_threshold # The amount of error in inches allowed for a fuel to keep the id

        self.init()

    def init(self):
        for i, fuel in enumerate(self.fuel_list):
            fuel.set_id(i)

    def update(self, delta_time: float):
        for fuel in self.fuel_list:
            predicted_x, predicted_y = fuel.predict_future(delta_time)

            for fuel in self.fuel_list:
                if fuel.id == fuel.id:
                    continue

                distance = spatial.euclidean((fuel.get_position(), (predicted_x, predicted_y)))
                if distance <= self.distance_threshold:
                    fuel.set_id(fuel.id)
                    break