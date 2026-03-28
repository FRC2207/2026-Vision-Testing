import numpy as np
import json
import VisionCoreConfig

CONFIG = VisionCoreConfig("Simplified/config.json")

MODE = "test"
# "game" for game
# "test" for testing with robot or sim (change network tables IP)
# "debug" for home stuff

if MODE == "game":
    CONFIG.set("debug_mode", False)
    CONFIG.set("app_mode", False)
    CONFIG.set("use_network_tables", True)
    CONFIG.set("network_tables_ip", "10.22.7.2")
elif MODE == "test":
    CONFIG.set("debug_mode", True)
    CONFIG.set("app_mode", True)
    CONFIG.set("use_network_tables", True)
    CONFIG.set("network_tables_ip", "192.168.1.166")
else:
    CONFIG.set("debug_mode", True)
    CONFIG.set("app_mode", True)
    CONFIG.set("use_network_tables", False)
    CONFIG.set("network_tables_ip", "")