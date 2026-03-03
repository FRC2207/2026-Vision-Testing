from Classes.Camera import Camera
import numpy as np

class MultipleCameraHandler:
    def __init__(self, cameras: list[Camera]):
        self.cameras = cameras
        self.model = cameras[0].model
        self.last_mosaic = None

    def get_robot_relative_points(self):
        tiles = []
        for cam in self.cameras:
            ret, frame = cam.cap.read()
            if not ret:
                frame = np.zeros((480, 640, 3), np.uint8)
            tiles.append(cv2.resize(frame, (320, 320)))

        top = np.hstack((tiles[0], tiles[1]))
        bottom = np.hstack((tiles[2], tiles[3]))
        self.last_mosaic = np.vstack((top, bottom))

        results = self.model.predict(self.last_mosaic)
        
        map_points = []
        for box in results.boxes:
            x1, y1, x2, y2 = box.xyxy
            if box.conf < self.cameras[0].min_confidence:
                continue

            cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
            
            col = 1 if cx > 320 else 0
            row = 1 if cy > 320 else 0
            idx = row * 2 + col
            
            if idx < len(self.cameras):
                cam = self.cameras[idx]
                
                tile_cx = cx - (col * 320)
                tile_cy = cy - (row * 320)
                tile_w = x2 - x1
                tile_h = y2 - y1

                point = cam.calculate_from_mosaic(tile_cx, tile_cy, tile_w, tile_h)
                if point is not None:
                    map_points.append(point)

        return np.array(map_points) if map_points else np.empty((0, 2))