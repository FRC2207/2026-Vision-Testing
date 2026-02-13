from networktables import NetworkTables
import time

class NetworkTableHandler:
    def __init__(self, ip: str):
        self.ip = ip
        # NetworkTables.initialize(server=self.ip)

        # self.connected = False

        # # Wait for a connection (optional, but recommended)
        # while not NetworkTables.isConnected():
        #     time.sleep(0.1)
        # self.connected = True

    def send_data(self, data: int|str|bool, data_name: str, table_name: str):
        # table = NetworkTables.getTable(table_name)
        # while True:
        #     try:
        #         if type(data) == int:
        #             table.putNumber(data_name, data)
        #         elif type(data) == str:
        #             table.putString(data_name, data)
        #         else:
        #             table.putBoolean(data_name, data)
        #     except KeyboardInterrupt:
        #         break
        pass