import time
import threading
from flask import jsonify, Response, request

class HealthReporter:
    def __init__(self, flask_app):
        self._lock = threading.Lock()
        self._camera = None
        self._network_handler = None

        self._fps: float = 0.0
        self._vision_s: float = 0.0
        self._detections: int = 0
        self._last_tick: float = time.perf_counter()
        self._uptime_start: float = time.perf_counter()
        self._loop_count: int = 0

        flask_app.add_url_rule("/health", "health", self._health_route)

    def set_camera(self, camera):
        self._camera = camera

    def set_network_handler(self, network_handler):
        self._network_handler = network_handler

    def tick(self, fps: float, vision_s: float, detections: int):
        with self._lock:
            self._fps = round(fps, 1)
            self._vision_s = round(vision_s * 1000, 2)
            self._detections = detections
            self._last_tick = time.perf_counter()
            self._loop_count += 1

    def _build_payload(self):
        now = time.perf_counter()

        with self._lock:
            fps = self._fps
            vision_ms = self._vision_s
            detections = self._detections
            last_tick = self._last_tick
            loop_count = self._loop_count

        stale_s = round(now - last_tick, 2)
        uptime_s = round(now - self._uptime_start, 1)

        # Camera
        camera_ok = False
        frame_age_ms = None
        if self._camera is not None:
            try:
                age = self._camera.get_frame_age()
                frame_age_ms = round(age * 1000, 1)
                camera_ok = age < 1.0
            except Exception:
                camera_ok = False

        # NetworkTables
        nt_connected = None
        if self._network_handler is not None:
            try:
                nt_connected = self._network_handler.isConnected()
            except Exception:
                nt_connected = False

        healthy = (
            stale_s < 2.0
            and camera_ok
            and (nt_connected is None or nt_connected)
        )

        payload = {
            "status": "ok" if healthy else "degraded",
            "uptime_s": uptime_s,
            "loop_count": loop_count,
            "loop_stale_s": stale_s,
            "fps": fps,
            "vision_ms": vision_ms,
            "detections": detections,
            "camera": {
                "ok": camera_ok,
                "frame_age_ms": frame_age_ms,
            },
            "network_tables": {
                "enabled": self._network_handler is not None,
                "connected": nt_connected,
            },
        }

        return payload, healthy

    def _render_html(self, payload):
        status_color = "ok" if payload["status"] == "ok" else "bad"

        return f"""
        <html>
        <head>
            <title>System Health</title>
            <meta http-equiv="refresh" content="1"> <!-- auto refresh -->
            <style>
                body {{ font-family: Arial; background: #111; color: #eee; }}
                .card {{ padding: 20px; margin: 20px; border-radius: 10px; background: #222; }}
                .ok {{ color: #4caf50; }}
                .bad {{ color: #f44336; }}
            </style>
        </head>
        <body>
            <h1>System Health</h1>

            <div class="card">
                <h2>Status: <span class="{status_color}">
                    {payload["status"].upper()}
                </span></h2>

                <p><b>FPS:</b> {payload["fps"]}</p>
                <p><b>Inference:</b> {payload["vision_ms"]} ms</p>
                <p><b>Detections:</b> {payload["detections"]}</p>
                <p><b>Loop stale:</b> {payload["loop_stale_s"]} s</p>
                <p><b>Uptime:</b> {payload["uptime_s"]} s</p>
            </div>

            <div class="card">
                <h3>Camera</h3>
                <p>Status: {"OK" if payload["camera"]["ok"] else "BAD"}</p>
                <p>Frame age: {payload["camera"]["frame_age_ms"]} ms</p>
            </div>

            <div class="card">
                <h3>NetworkTables</h3>
                <p>Enabled: {payload["network_tables"]["enabled"]}</p>
                <p>Connected: {payload["network_tables"]["connected"]}</p>
            </div>
        </body>
        </html>
        """

    def _health_route(self):
        payload, healthy = self._build_payload()

        # Browser is HTML, everything else is raw JSON
        wants_html = "text/html" in request.headers.get("Accept", "")

        if wants_html:
            return Response(self._render_html(payload), mimetype="text/html")

        return jsonify(payload), (200 if healthy else 503)