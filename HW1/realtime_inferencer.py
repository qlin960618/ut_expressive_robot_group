import os
import traceback
import time
from typing import Callable

import numpy as np
import cv2
import logging

import mediapipe as mp
from mediapipe.tasks.python import vision as mp_vision

logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

class Inferencer:
    def __init__(self, dev, detector, draw_function: Callable, **kwargs):
        self.cap = cv2.VideoCapture(dev)
        self.dev = dev

        if not self.cap.isOpened():
            logger.error(f"Failed to open device {dev}")
            raise RuntimeError(f"Failed to open device {dev}")

        self.vid_dim = (int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
        logger.info(f"Video dimension: {self.vid_dim}")

        self.detector = detector
        self.draw_function = draw_function

        self.show_original = kwargs.get("show_original", False)
        self.show_fps = kwargs.get("show_fps", False)
        self.record_path = kwargs.get("record_path", None)

        if self.record_path is not None:
            self.fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            os.makedirs(os.path.dirname(self.record_path), exist_ok=True)
            self.recorder = cv2.VideoWriter(self.record_path, self.fourcc, 15.0, self.vid_dim)
        else:
            self.recorder = None

    def start_loop(self):
        logger.info("start capturing")
        ret = True
        start_time = time.time()
        while ret:
            ret, frame = self.cap.read()
            if not ret:
                logger.error(f"Failed to read from device {self.dev}")
                return None
            if frame is not None:
                if self.show_original:
                    cv2.imshow(f"original", frame)

                detection_result = self.detector.detect(frame)

                annotated_image = self.draw_function(frame, detection_result)

                if annotated_image is not None:
                    cv2.imshow(f"labeled", annotated_image)

                if self.recorder is not None and annotated_image is not None:
                    self.recorder.write(annotated_image)

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                if self.show_fps:
                    time_now = time.time()
                    fps = 1.0 / (time_now - start_time)
                    start_time = time_now
                    logger.info(f"fps: {fps:.2f}")
            else:
                logger.error(f"Failed to read from device {self.dev}, Frame is None")

    def __del__(self):
        self.cap.release()
        if self.recorder is not None:
            self.recorder.release()
        cv2.destroyAllWindows()


