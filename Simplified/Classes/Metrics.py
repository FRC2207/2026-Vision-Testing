from collections import deque
import logging

class Metrics:
    def __init__(self, window: int = 30, log_every: int = 30):
        self.window    = window
        self.log_every = log_every
        self._itr      = 0
        self._data: dict[str, deque] = {}
        self.logger = logging.getLogger(__name__)

    def record(self, **kwargs: float):
        for key, val in kwargs.items():
            if val is None:
                continue
            if key not in self._data:
                self._data[key] = deque(maxlen=self.window)
            self._data[key].append(val)
 
    def avg(self, key: str) -> float | None:
        if key not in self._data or len(self._data[key]) == 0:
            return None
        return sum(self._data[key]) / len(self._data[key])
 
    def tick(self):
        self._itr += 1
        if self._itr % self.log_every == 0:
            self._log()
 
    def _fmt(self, key: str, unit: str = "ms", scale: float = 1000.0) -> str:
        v = self.avg(key)
        return f"{v * scale:.1f}{unit}" if v is not None else "n/a"
 
    def _log(self):
        loop_ms    = self._fmt("loop_s")
        vision_ms  = self._fmt("vision_s")
        cam_lag_ms = self._fmt("camera_lag_s")
        flask_ms   = self._fmt("flask_s")
        fps        = self.avg("loop_s")
        fps_str    = f"{1/fps:.1f}" if fps else "n/a"
 
        self.logger.info(
            f"[Metrics @{self._itr}] "
            f"loop={loop_ms}  fps={fps_str}  "
            f"vision={vision_ms}  "
            f"cam_lag={cam_lag_ms}  "
            f"flask={flask_ms}"
        )