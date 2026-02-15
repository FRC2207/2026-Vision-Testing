from networktables import NetworkTables
import time
import numpy as np

class NetworkTableHandler:
    def __init__(self, ip: str):
        self.ip = ip
        NetworkTables.initialize(server=self.ip)

        self.connected = True

        print("[Custom PathPlanner] Initializing NetworkTables connection...")
        # Wait for a connection (optional, but recommended)
        i = 0
        while not NetworkTables.isConnected():
            time.sleep(0.1)
            i += 1

            if (i > 60):  # Wait for 60 seconds max
                print("[Custom PathPlanner] Warning: NetworkTables connection timed out after 60 seconds.")
                self.connected = False
                break

    def send_data(self, data, data_name: str, table_name: str):
        if self.connected:
            try:
                table = NetworkTables.getTable(table_name)
                if isinstance(data, (list, np.ndarray)):
                    # Convert to list of floats
                    data_list = [float(x) for sublist in data for x in sublist] if isinstance(data, np.ndarray) else data
                    table.putNumberArray(data_name, data_list)
                elif isinstance(data, (int, float)):
                    table.putNumber(data_name, data)
                elif isinstance(data, str):
                    table.putString(data_name, data)
                else:
                    table.putBoolean(data_name, data)
            except Exception as e:
                print(f"[NetworkTables Error] {e}")
        else:
            print("[NetworkTables Warning] Not connected. Data not sent.")