from networktables import NetworkTables
import time
import numpy as np
import logging

class NetworkTableHandler:
    def __init__(self, ip: str):
        self.ip = ip
        self.logger = logging.getLogger(__name__)
        NetworkTables.initialize(server=self.ip)

        self.connected = True

        self.logger.info("Initializing NetworkTables connection...")
        # Wait for a connection (optional, but recommended)
        i = 0
        while not NetworkTables.isConnected():
            time.sleep(0.1)
            i += 1

            if (i > 600):  # Wait for 60 seconds max
                self.logger.warning(f"NetworkTables connection timed out after 60 seconds (1 minute for those who don't know math).")
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
                self.logger.error(f"{e}")
        else:
            self.logger.warning("Not connected. Data not sent.")