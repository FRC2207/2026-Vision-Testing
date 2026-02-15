import cv2
from flask import Flask, Response
import threading


class CameraApp:
    def __init__(self):
        self.app = Flask(__name__)
        self.current_frame = None
        self.frame_lock = threading.Lock()
        
        self.app.add_url_rule('/video_feed', 'video_feed', self.video_feed)
        self.app.add_url_rule('/', 'index', self.index)
    
    def set_frame(self, frame):
        with self.frame_lock:
            self.current_frame = frame
    
    def generate_frames(self):
        while True:
            with self.frame_lock:
                if self.current_frame is None:
                    continue
                frame = self.current_frame.copy()
            
            ret, buffer = cv2.imencode('.jpg', frame)
            frame_bytes = buffer.tobytes()
            
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
    
    def video_feed(self):
        return Response(self.generate_frames(),
                        mimetype='multipart/x-mixed-replace; boundary=frame')
    
    def index(self):
        return """
        <!DOCTYPE html>
        <html>
        <body>
            <img src="/video_feed" alt="Video Stream">
        </body>
        </html>
        """
    
    def run(self, host='0.0.0.0', port=5000, debug=False, use_reloader=False):
        self.app.run(host=host, port=port, debug=debug, use_reloader=False)
