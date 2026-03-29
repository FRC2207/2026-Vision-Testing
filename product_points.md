# VisionCore — Current Skill Showcase

> Robotics vision coprocessor targeting FRC competition use.
> Built for Orange Pi 5 Pro (6 TOPS NPU) + USB camera.

---

## Architecture Overview

```
Camera (threaded reader + preproc worker)
    │
    ├── _reader()          ← background thread, always grabs latest frame
    ├── _preprocess_worker() ← background thread, letterboxes into pre-alloc buffer
    │
    └── YoloWrapper
            ├── RKNN backend  (NPU — Orange Pi / Rockchip)
            └── Ultralytics   (CPU/GPU — laptop dev)

Camera.run()
    └── robot-relative (x, y) positions
            │
            └── FuelTracker.update(fuel_list, robot_pose)
                    └── field-relative tracked positions
                            │
                            └── NetworkTableHandler  →  Robot (WPILib)
                            └── CameraApp (Flask)    →  Browser dashboard
```

---

## What's Working

### Dual-backend inference
- **RKNN** path uses pre-allocated input buffers (no malloc per frame), pipelined preprocessing concurrent with the previous inference
- **Ultralytics** path (ONNX / OpenVINO / .pt) auto-selected when RKNN isn't available — same API surface
- Output format **auto-detected** on first inference: `end2end` (6-col) vs `no_nms` (5-col)
- Confidence activation **auto-detected**: sigmoid applied only if raw logits detected (< -0.1 or > 1.1)

### Coordinate pipeline
```
Pixel (cx, cy)
    → horizontal_angle  = atan(pixel_offset_x / focal_length_px)
    → ground_distance   = camera_height / tan(angle_to_ground - pitch)  [if height > 0]
                        = distance_LOS * cos(pitch)                      [flat]
    → (forward, lateral) in camera frame
    → rotate by camera_bot_relative_yaw
    → offset by (camera_x, camera_y)
    → scale by unit conversion (inch / meter / foot / cm)
    → robot-relative (x, y)
```

### Field-relative tracking (FuelTracker)
- New detections: `field_pos = robot_pos + robot_relative_detection`
- Existing tracked objects: stay fixed in field space (no drift)
- Deduplication: L2 distance threshold, resets TTL on match
- TTL-based destruction: objects expire if not re-observed

### Config system (VisionCoreConfig)
- Deep-merge JSON load with typed defaults
- `__getitem__`, `__call__`, `__getattr__` all work: `config["key"]`, `config("key")`, `config.key`
- Per-camera calibration: FOV, focal length, height, pitch, yaw, position offsets
- Runtime mode switching: `game` / `test` / `debug` changes NT + logging in one place

---

## New Additions (This Session)

### `/health` endpoint (`Classes/health.py`)
HTTP health check at `GET /health` — returns JSON, `200 OK` or `503 Degraded`.

```json
{
  "status": "ok",
  "uptime_s": 142.3,
  "loop_count": 4201,
  "loop_stale_s": 0.02,
  "fps": 28.4,
  "vision_inference_ms": 12.1,
  "detections_last_frame": 3,
  "camera": { "ok": true, "frame_age_ms": 18.2 },
  "network_tables": { "enabled": true, "connected": true }
}
```

Wire-up (3 lines in `solo_game_loop.py`):
```python
from Classes.health import HealthReporter
health = HealthReporter(camera_app.app)
health.set_camera(camera)
health.set_network_handler(network_handler)
# in loop:
health.tick(fps=1/loop_s, vision_s=vision_s, detections=len(fuel_list))
```

---

### Auto camera detection (`Classes/camera_detect.py`)
Replaces the hardcoded `"Leveno Laptop"` string. On Linux uses `v4l2-ctl` to read
physical device names and fuzzy-matches against `config.json` camera names.
Falls back to OpenCV index probing on Windows/macOS.

```python
from Classes.camera_detect import resolve_camera_config

cam_config = resolve_camera_config(constants.CONFIG)
# or with a CLI hint:
cam_config = resolve_camera_config(constants.CONFIG, preferred_name="Arducam")
```

Match scoring: counts how many words from the config name appear in the physical device name.
`"Microsoft Cinema"` config → matches `"Microsoft LifeCam Cinema (usb-3.1-...)"` with score 2.

## Comparison vs. Commercial Systems

| | **VisionCore** | **Limelight 4** | **PhotonVision** | **OAK-D Lite** |
|---|---|---|---|---|
| Price | ~$80 (OPi5 Pro + cam) | ~$400 | Free (DIY HW) | ~$150 |
| NPU inference | ✅ 6 TOPS | ✅ | ❌ CPU only | ✅ (Myriad X) |
| Custom models | ✅ | ✅ (paid tier) | ✅ | ✅ |
| AprilTag | ❌ (planned) | ✅ | ✅ | ✅ |
| Open source | ✅ | ❌ | ✅ | ✅ |
| NT out-of-box | ✅ | ✅ | ✅ | ❌ |
| FRC community | 🔨 building | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐ |
| Plug-and-play | 🔨 building | ✅ | Partial | ❌ |

**Key differentiator:** Fastest neural inference per dollar in FRC. PhotonVision on the same
Orange Pi hardware runs inference on CPU. VisionCore uses the NPU — that's the number to benchmark
and lead with.