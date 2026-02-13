import cv2
import math
import numpy as np
from ultralytics import YOLO
import requests
import time

class Camera:
    def __init__(self, source: int|str):
        self.source = source

        self.cap = cv2.VideoCapture(self.source)
        time.sleep(0.5)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1) 

        if not self.cap.isOpened():
            raise ValueError(f"Cannot open source {self.source}")

    def get_frame(self):
        ret, frame = self.cap.read()
        if not ret:
            raise ValueError(f"Failed to retrieve frame from: {self.source}")
        return frame
                    
    def run(self):
        frame = self.get_frame()
        # Show it with cv2
        cv2.imshow("YOLO Detections", frame)
        cv2.waitKey(1)
    
    def destroy(self):
        self.cap.release()
        cv2.destroyAllWindows()
        

# Camera class
camera = Camera(
    "http://photon1.local:1181/stream.mjpg"
)


while True:
    camera.run()