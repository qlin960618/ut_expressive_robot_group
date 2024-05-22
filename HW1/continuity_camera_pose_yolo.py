import traceback

import numpy as np
from PIL import Image
import cv2
import logging
from ultralytics import YOLO

# Custom tools
from realtime_inferencer import Inferencer

# wget -O https://github.com/ultralytics/assets/releases/download/v8.2.0/yolov8n-pose.pt

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


def draw_landmarks_on_image(rgb_image, detection_result):
    annotated_frame = detection_result[0].plot()

    return annotated_frame


class DetectorDummy:
    def __init__(self, model):
        self.model = model

    def detect(self, image):
        results = self.model(image)
        return results


device = 1

if __name__ == "__main__":
    logger.info("loading YOLO model")

    # Load a model
    yolo_model = YOLO('tmp/yolov8n-pose.pt')  # load an official model
    yolo_model.to("mps")
    logger.info("yolo loaded")

    inferencer = Inferencer(device, DetectorDummy(model=yolo_model), draw_landmarks_on_image,
                            show_original=True, show_fps=True,
                            record_path="tmp/pose_landmarker_yolo_test2.mp4")
    try:
        logger.info("starting loop")
        inferencer.start_loop()

    except KeyboardInterrupt as e:
        logger.warning("exit on KeyboardInterrupt")
    except Exception as e:
        logger.error(f"exit on {e}")
        traceback.print_exc()
    finally:

        cv2.destroyAllWindows()
