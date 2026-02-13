import numpy as np

class Fuel:
    def __init__(self, x: int, y: int, id: int=-1):
        self.x = x
        self.y = y
        self.id = id

    def get_position(self):
        return np.array([self.x, self.y])
    
    def get_id(self):
        return self.id
    
    def set_id(self, id: int):
        self.id = id