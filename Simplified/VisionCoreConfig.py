import json
import logging

class VisionCoreConfig:
    def __init__(self, file_path: str = None):
        self.logger = logging.getLogger(__name__)

        self.default_config = {
            "unit": "meter",
            "dbscan": {"elipson": 0, "min_samples": 0},
            "distance_threshold": -1,
            "vision_model_input_size": [640, 640],
            "vision_model_file_path": "model.pt",
            "network_tables_ip": "10.22.7.2",
            "use_network_tables": True,
            "app_mode": False,
            "debug_mode": False,
            "camera_configs": {
                "default": {
                    "name": "default",
                    "x": 0, "y": 0, "height": 0, "pitch": 0, "yaw": 0,
                    "grayscale": False,
                    "fps_cap": -1,
                    "calibration": {"size": 0, "distance": 0, "game_piece_size": 0, "fov": 0},
                    "source": "/dev/video0",
                    "subsystem": "field"
                },
            },
            "vision_model": {
                "quantized": False,
                "file_path": "model.pt",
                "input_size": [640, 640]
            }
        }
        self.config = self.default_config.copy()

        if file_path:
            self.load_from_file(file_path)

        self.camera_configs = {}
        for cam_name, cam_config in self.config["camera_configs"].items():
            self.camera_configs[cam_name] = VisionCoreCameraConfig(cam_config)
        
    def camera_config(self, cam_name: str):
        return self.camera_configs.get(cam_name, {})

    def load_from_file(self, file_path: str):
        try:
            with open(file_path, 'r') as f:
                data = json.load(f)
                self._update_config(data)
        except Exception as e:
            self.logger.warning(f"Failed to load config from {file_path}: {e}")
            self.logger.info("Using default configuration.")

    def get(self, *keys):
        val = self.config
        try:
            for key in keys:
                val = val[key]  # Move deeper
            return val
        except (KeyError, TypeError):
            self.logger.warning(f"Key path {keys} not found.")
            return None

    def set(self, value, *keys):
        if not keys: return
        
        target = self.config
        for key in keys[:-1]:
            # Ensure the path exists as a dictionary
            target = target.setdefault(key, {})
        target[keys[-1]] = value

    def _update_config(self, data: dict, current_dict=None):
        if current_dict is None:
            current_dict = self.config

        for key, value in data.items():
            if isinstance(value, dict) and key in current_dict and isinstance(current_dict[key], dict):
                self._update_config(value, current_dict[key])
            else:
                current_dict[key] = value

    # Keep these for backwards comapability and ease
    def __getitem__(self, args):
        if isinstance(args, tuple):
            return self.get(*args)
        return self.get(args)

    def __call__(self, *keys):
        return self.get(*keys)
    
    def __getattr__(self, item):
        return self.get(item)
    
class VisionCoreCameraConfig:
    def __init__(self, config_dict: dict = None):
        self.defaults = {
            "name": "default",
            "x": 0, "y": 0, "height": 0, "pitch": 0, "yaw": 0,
            "grayscale": False,
            "fps_cap": -1,
            "calibration": {"size": 0, "distance": 0, "game_piece_size": 0, "fov": 0},
            "source": "/dev/video0",
            "subsystem": "field"
        }
        
        self.data = self.defaults.copy()
        if config_dict:
            self.data.update(config_dict)

    def __getitem__(self, key):
        return self.data.get(key)

    def get(self, key, default=None):
        return self.data.get(key, default)