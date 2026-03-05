import ntcore
import logging
import numpy as np
from Classes.Fuel import Fuel

class NetworkTableHandler:
    def __init__(self, ip: str):
        self.ip = ip
        self.logger = logging.getLogger(__name__)
        self.inst = ntcore.NetworkTableInstance.getDefault()
        self.inst.startClient4("CustomVisionStuff")
        self.inst.setServerTeam(2207)
        self.connected = True
        
        self._subscribers = {}
        self._tables = {}

    def _get_table(self, table_name: str):
        if table_name not in self._tables:
            self._tables[table_name] = self.inst.getTable(table_name)
        return self._tables[table_name]

    def send_data(self, data, data_name: str, table_name: str):
        if not self.connected:
            return

        table = self._get_table(table_name)
        
        if isinstance(data, list) and len(data) > 0 and isinstance(data[0], Fuel):
            positions = [f.get_position() for f in data]
            table.putNumberArray(f"{data_name}_X", [p[0] for p in positions])
            table.putNumberArray(f"{data_name}_Y", [p[1] for p in positions])
        
        elif isinstance(data, (int, float)):
            table.putNumber(data_name, float(data))
        elif isinstance(data, str):
            table.putString(data_name, data)
        elif isinstance(data, bool):
            table.putBoolean(data_name, data)

    def get_data(self, data_type, data_name: str, table_name: str):
        if not self.connected:
            return [0.0, 0.0]

        # Unique key for this specific data stream
        sub_key = f"{table_name}/{data_name}"
        
        if sub_key not in self._subscribers:
            table = self._get_table(table_name)
            if isinstance(data_type, (list, np.ndarray)):
                self._subscribers[sub_key] = table.getDoubleArrayTopic(data_name).subscribe([])
            elif isinstance(data_type, (int, float)):
                self._subscribers[sub_key] = table.getDoubleTopic(data_name).subscribe(0.0)
            elif isinstance(data_type, str):
                self._subscribers[sub_key] = table.getStringTopic(data_name).subscribe("")
        
        return self._subscribers[sub_key].get()
