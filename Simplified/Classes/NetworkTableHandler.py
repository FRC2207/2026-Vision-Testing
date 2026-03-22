import ntcore
import logging
import numpy as np
from Classes.Fuel import Fuel
import time
import dataclasses
import wpiutil.wpistruct
from ntcore import NetworkTableInstance

@wpiutil.wpistruct.make_wpistruct(name="Fuel")
@dataclasses.dataclass
class FuelStruct:
    x: float
    y: float

class NetworkTableHandler:
    def __init__(self, ip: str):
        self.ip = ip
        self.logger = logging.getLogger(__name__)
        self.inst = ntcore.NetworkTableInstance.getDefault()
        self.inst.setServer(self.ip)
        self.inst.startClient4("CustomVisionStuff")

        i = 0
        while (not self.inst.isConnected()) and (i < 15):
            self.logger.warning("Network tables not connected, attempting to connect.")

            time.sleep(1)
            i += 1

        if i >= 15:
            self.logger.error(f"Network tables could not connect after 15 seconds")

        self._subscribers = {}
        self._tables = {}

    def _get_table(self, table_name: str):
        if table_name not in self._tables:
            self._tables[table_name] = self.inst.getTable(table_name)
        return self._tables[table_name]
    
    def send_fuel_list(self, fuels: list[Fuel], data_name: str="fuel_data", table_name: str="VisionData"):
        try:
            if not self.inst.isConnected():
                return

            table = self._get_table(table_name)
            
            struct_list = []
            for f in fuels:
                pos = f.get_position_normally()
                struct_list.append(FuelStruct(x=float(pos[0]), y=float(pos[1])))

            pub_key = f"pub/{table_name}/{data_name}"
            if pub_key not in self._subscribers:
                self._subscribers[pub_key] = table.getStructArrayTopic(data_name, FuelStruct).publish()
            
            self._subscribers[pub_key].set(struct_list)
            table.putNumber("timestamp_ms", time.time() * 1000)
            self.inst.flush()
            
            self.logger.info(f"Sent {len(struct_list)} fuels via StructArray")
        except Exception as e:
            self.logger.error(f"Failed to send fuel structs: {e}")

    def get_data(self, data_type, data_name: str, table_name: str):
        if not self.inst.isConnected():
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
