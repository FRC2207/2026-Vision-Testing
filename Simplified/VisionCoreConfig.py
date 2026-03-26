import json
import logging

class VisionCoreConfig:
    def __init__(self, file_path: str = None):
        self.default_config = {
            "unit": "meter",
            "dbscan": {
                "elipson": 0,
                "min_samples": 0
            },
            "distance_threshold": -1,
            "vision_model_input_size": [640, 640],
            "vision_model_file_path": "model.pt",
            "network_tables_ip": "10.22.7.2",
            "use_network_tables": True,
            "app_mode": False,
            "debug_mode": False,
            "camera_configs": {

            }
        }

        self.config = self.default_config.copy()

        if file_path:
            self.load_from_file(file_path)

        self.logger = logging.getLogger(__name__)

    def set(self, key: str, value):
        if key in self.default_config:
            self.config[key] = value
        else:
            self.logger.warning(f"Ignoring attempt to set unknown config key: '{key}'")

    def load_from_file(self, file_path: str):
        with open(file_path, "r") as file:
            data = json.load(file)

        self._update_config(data)

    def load_from_file(self, file_path: str):
            try:
                with open(file_path, 'r') as file:
                    data = json.load(file)
                    self._update_config(data)
            except FileNotFoundError:
                self.logger.error(f"Config file not found: {file_path}")

    def _update_config(self, data: dict):
        for key, value in data.items():
            if key in self.default_config:
                self.config[key] = value
            else:
                self.logger.warning(f"Ignoring unknown config key: '{key}'")
    
    def __getitem__(self, key: str):
        return self.config[key]