from flask import Flask, Response, render_template_string
import cv2
import threading
import logging

PAGE = """
<!DOCTYPE html>
<html>
<head><title>Camera Feed</title></head>
<body style="margin:0;background:#111">
<img id="feed" src="/video_feed" style="display:block"/>
<script>
  fetch('/dimensions')
    .then(r => r.json())
    .then(d => {
      const img = document.getElementById('feed');
      img.style.width  = d.width  + 'px';
      img.style.height = d.height + 'px';
    });
</script>
</body>
</html>
"""

log = logging.getLogger('werkzeug')
log.setLevel(logging.ERROR)

class CameraApp:
    def __init__(self):
        self.app    = Flask(__name__)
        self.frame  = None
        self.lock   = threading.Lock()
        self.width  = 640 # Updates once first frame arrives
        self.height = 480 # Updates once first frame arrives
        self.logger = logging.getLogger(__name__)

        self.app.add_url_rule('/',            'index',      self._index)
        self.app.add_url_rule('/video_feed',  'video_feed', self._video_feed)
        self.app.add_url_rule('/dimensions',  'dimensions', self._dimensions)

    def set_frame(self, frame):
        if frame is None:
            return
        with self.lock:
            self.frame  = frame
            self.height, self.width = frame.shape[:2]

    def run(self, host='0.0.0.0', port=5000):
        self.app.run(host=host, port=port, threaded=True)

    def _index(self):
        return render_template_string(PAGE)

    def _dimensions(self):
        from flask import jsonify
        return jsonify(width=self.width, height=self.height)

    def _video_feed(self):
        return Response(
            self._generate(),
            mimetype='multipart/x-mixed-replace; boundary=frame'
        )

    def _generate(self):
        import time
        while True:
            with self.lock:
                frame = self.frame

            if frame is None:
                time.sleep(0.05)
                continue

            ok, buf = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
            if not ok:
                continue

            yield (
                b'--frame\r\n'
                b'Content-Type: image/jpeg\r\n\r\n'
                + buf.tobytes()
                + b'\r\n'
            )