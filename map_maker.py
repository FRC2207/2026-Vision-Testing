import matplotlib.pyplot as plt
import json

class FRCDataCollector:
    def __init__(self, filename="ball_layout.json"):
        self.filename = filename
        self.points = []
        
        # Setup Plot
        self.fig, self.ax = plt.subplots(figsize=(9, 8))
        self.ax.set_title("L-Click: Add Ball | R-Click: Clear | Close: Save")
        
        # Lock ranges -50 to 50 and y to 0 to 100
        self.ax.set_xlim(-50, 50)
        self.ax.set_ylim(0, 100)
        self.ax.grid(True, linestyle='--', alpha=0.6)
        
        self.plot_handle, = self.ax.plot([], [], 'bo', markersize=8) # Current points
        
        # Connect events
        self.cid = self.fig.canvas.mpl_connect('button_press_event', self.onclick)

    def onclick(self, event):
        if event.inaxes != self.ax:
            return
            
        if event.button == 1: # Left Click - Add Point
            self.points.append({"x": round(event.xdata, 2), "y": round(event.ydata, 2)})
            print(f"Added: ({event.xdata:.1f}, {event.ydata:.1f})")
            
        elif event.button == 3: # Right Click - Clear All
            self.points = []
            print("Cleared all points.")

        self.update_plot()

    def update_plot(self):
        if self.points:
            x_vals = [p['x'] for p in self.points]
            y_vals = [p['y'] for p in self.points]
            self.plot_handle.set_data(x_vals, y_vals)
        else:
            self.plot_handle.set_data([], [])
        self.fig.canvas.draw()

    def save_to_json(self):
        output = {
            "metadata": {"range": "-50-50x, 0-100y", "count": len(self.points)},
            "points": self.points
        }
        
        with open(self.filename, 'w') as f:
            json.dump(output, f, indent=4)
        print(f"\n--- SUCCESS ---")
        print(f"Saved {len(self.points)} points to {self.filename}")

collector = FRCDataCollector("ball_layout.json")
plt.show() 

# Triggers when window is closed
collector.save_to_json()