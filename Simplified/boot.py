import json
import threading
import time
from pathlib import Path
import os
import cv2
from flask import Flask, jsonify, render_template_string, request, Response
import logging

logging.basicConfig(level=logging.INFO)

CONFIG_FILE = Path("Simplified/config.json")
SHUTDOWN_FLAG = threading.Event()

_captures: dict[int, cv2.VideoCapture] = {}
_cap_lock = threading.Lock()

def get_cap(index: int) -> cv2.VideoCapture:
    with _cap_lock:
        if index not in _captures or not _captures[index].isOpened():
            _captures[index] = cv2.VideoCapture(index)
        return _captures[index]

def release_all():
    with _cap_lock:
        for cap in _captures.values():
            cap.release()
        _captures.clear()

def discover_cameras(max_index=8):
    devnull = open(os.devnull, "w")
    saved_stderr = os.dup(2)
    os.dup2(devnull.fileno(), 2)
    cameras = []
    for idx in range(max_index):
        cap = cv2.VideoCapture(idx)
        if not cap.isOpened():
            cap.release()
            break

        cameras.append({
            "index":      idx,
            "name":       f"Camera {idx}",
            "width":      int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            "height":     int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            "brightness": round(cap.get(cv2.CAP_PROP_BRIGHTNESS), 2),
            "contrast":   round(cap.get(cv2.CAP_PROP_CONTRAST), 2),
            "exposure":   round(cap.get(cv2.CAP_PROP_EXPOSURE), 2),
        })
        cap.release()

    return cameras

app = Flask(__name__)

@app.route("/")
def index():
    return render_template_string(Path("Simplified/HTML/boot.html").read_text())

@app.route("/api/cameras")
def api_cameras():
    return jsonify(discover_cameras())

@app.route("/api/update_setting", methods=["POST"])
def api_update_setting():
    data = request.get_json(force=True)
    idx = data.get("index")
    if idx is None:
        return jsonify({"ok": False, "error": "Missing index"}), 400

    cap = get_cap(idx)
    if "brightness" in data:
        cap.set(cv2.CAP_PROP_BRIGHTNESS, float(data["brightness"]))
    if "contrast" in data:
        cap.set(cv2.CAP_PROP_CONTRAST, float(data["contrast"]))
    if "exposure" in data:
        cap.set(cv2.CAP_PROP_EXPOSURE, float(data["exposure"]))

    return jsonify({"ok": True})

@app.route("/api/save", methods=["POST"])
def api_save():
    data = request.get_json(force=True)
    cameras_list = data.get("cameras", [])

    camera_configs = {}
    for cam in cameras_list:
        name = cam["name"]
        camera_configs[name] = {
            "name":      name,
            "x":         cam.get("x", 0),
            "y":         cam.get("y", 0),
            "height":    cam.get("height", 0),
            "pitch":     cam.get("pitch", 0),
            "yaw":       cam.get("yaw", 0),
            "grayscale": cam.get("grayscale", False),
            "fps_cap":   cam.get("fps_cap", 30),
            "calibration": {
                "size":            cam.get("cal_size", 0),
                "distance":        cam.get("cal_distance", 12),
                "game_piece_size": cam.get("cal_game_piece_size", 5.91),
                "fov":             cam.get("cal_fov", 60),
            },
            "source":    cam.get("source", cam.get("index", 0)),
            "subsystem": cam.get("subsystem", "field"),
        }

    full_config = {
        "camera_configs": camera_configs
    }

    with open(CONFIG_FILE, "r") as f:
        existing = json.load(f)

    existing.update(full_config)

    CONFIG_FILE.write_text(json.dumps(existing, indent=4))
    return jsonify({"ok": True, "path": str(CONFIG_FILE.resolve())})

@app.route("/api/shutdown", methods=["POST"])
def api_shutdown():
    SHUTDOWN_FLAG.set()
    return jsonify({"ok": True})

def gen_frames(index: int):
    cap = get_cap(index)
    while True:
        ret, frame = cap.read()
        if not ret:
            time.sleep(0.05)
            continue
        _, buffer = cv2.imencode(".jpg", frame)
        yield (b"--frame\r\n"
               b"Content-Type: image/jpeg\r\n\r\n" + buffer.tobytes() + b"\r\n")
        time.sleep(1 / 30)

@app.route("/video_feed")
def video_feed():
    index = request.args.get("index", 0, type=int)
    return Response(gen_frames(index),
                    mimetype="multipart/x-mixed-replace; boundary=frame")

def main():
    host, port = "127.0.0.1", 5050
    threading.Thread(
        target=lambda: app.run(host=host, port=port, debug=False, use_reloader=False),
        daemon=True,
    ).start()

    logging.info(f"Open http://{host}:{port}/ to configure your cameras.")
    SHUTDOWN_FLAG.wait()
    release_all()
    logging.info(f"Done! Config saved to: {CONFIG_FILE.resolve()}")
    time.sleep(0.5)

if __name__ == "__main__":
    main()