import json
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.patches as patches

BALL_LAYOUT = "ball_layout.json" 
INTAKE_WIDTH = 10

def load_data(filename: str):
    with open(filename, 'r') as f:
        data_json = json.load(f)
    points = data_json.get("points", [])
    meta_data = data_json.get("metadata", {"xrange": [-50, 50], "yrange": [0, 100], "count": 0})
    # Convert list of dicts to numpy array
    coords = np.array([[p['x'], p['y']] for p in points])
    return coords, meta_data.get("xrange", [-50, 50]), meta_data.get("yrange", [0, 100])

fuel_positions, x_range, y_range = load_data(BALL_LAYOUT)

best_count = -1
best_grid = (0, 0)
grid_results = []

step_size = 1
x_coords = np.arange(x_range[0], x_range[1] - INTAKE_WIDTH, step_size)
y_coords = np.arange(y_range[0], y_range[1] - INTAKE_WIDTH, step_size)

for x in x_coords:
    for y in y_coords:
        x_min, x_max = x, x + INTAKE_WIDTH
        y_min, y_max = y, y + INTAKE_WIDTH
        
        in_x = (fuel_positions[:, 0] >= x_min) & (fuel_positions[:, 0] <= x_max)
        in_y = (fuel_positions[:, 1] >= y_min) & (fuel_positions[:, 1] <= y_max)
        count = np.sum(in_x & in_y)
        
        grid_results.append(((x, y), count))
        
        if count > best_count:
            best_count = count
            best_grid = (x, y)

print(f"Best Grid Found at: {best_grid} with {best_count} fuel")

fig, ax = plt.subplots(figsize=(8, 10))

ax.scatter(fuel_positions[:, 0], fuel_positions[:, 1], c='blue', label='Fuel Points', s=10)

rect = patches.Rectangle(best_grid, INTAKE_WIDTH, INTAKE_WIDTH, 
                         linewidth=2, facecolor='red', alpha=0.3, 
                         label=f'Best Intake ({best_count} fuel)')
ax.add_patch(rect)

ax.set_xlim(x_range)
ax.set_ylim(y_range)
ax.set_title(f'Robotics Intake Grid Search ({INTAKE_WIDTH}x{INTAKE_WIDTH})')
ax.set_xlabel('Field X')
ax.set_ylabel('Field Y')
ax.legend()
ax.grid(True)
plt.show()
