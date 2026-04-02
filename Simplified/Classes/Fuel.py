import numpy as np
import math
import time

class Fuel:
    def __init__(self, x: int, y: int, id: int=-1):
        self.x = x
        self.y = y
        self.id = id
        self.start_time = time.perf_counter()
        self.alive = 0

        self.destroyed = False
        self.alive_time = 0.2

    def relative_to(self, robot):
        self.x += robot[0]
        self.y += robot[1]

    def get_position(self):
        return np.array([self.x, self.y])
    
    def reset_time(self):
        self.start_time = time.perf_counter()
    
    def get_position_normally(self):
        return (self.x, self.y)
    
    def get_id(self):
        return self.id
    
    def set_id(self, id: int):
        self.id = id

    def update(self):
        self.alive = time.perf_counter() - self.start_time
        if (self.alive >= self.alive_time):
            self.destroyed = True

    def __str__(self):
        return f"Distance: {math.hypot(self.x, self.y)}, X: {self.x}, Y: {self.y}, Alive: {self.alive}"