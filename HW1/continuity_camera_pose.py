import traceback

from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2
import numpy as np
import cv2
import logging

import mediapipe as mp
from mediapipe import tasks
from mediapipe.tasks.python import vision as mp_vision

# Custom tools
from realtime_inferencer import Inferencer

# wget -O pose_landmarker.task -q https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_heavy/float16/1/pose_landmarker_heavy.task

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


def draw_landmarks_on_image(rgb_image, detection_result):
    pose_landmarks_list = detection_result.pose_landmarks
    annotated_image = np.copy(rgb_image)

    # Loop through the detected poses to visualize.
    for idx in range(len(pose_landmarks_list)):
        pose_landmarks = pose_landmarks_list[idx]

        # Draw the pose landmarks.
        pose_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
        pose_landmarks_proto.landmark.extend([
            landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y, z=landmark.z) for landmark in pose_landmarks
        ])
        solutions.drawing_utils.draw_landmarks(
            annotated_image,
            pose_landmarks_proto,
            solutions.pose.POSE_CONNECTIONS,
            solutions.drawing_styles.get_default_pose_landmarks_style())
    return annotated_image


device = 1

if __name__ == "__main__":
    logger.info("loading mediapipe")
    base_options = mp.tasks.BaseOptions(
        model_asset_path='tmp/pose_landmarker.task'
    )
    options = mp_vision.PoseLandmarkerOptions(
        base_options=base_options,
        output_segmentation_masks=True
    )

    detector = mp_vision.PoseLandmarker.create_from_options(options)
    logger.info("mediapipe loaded")

    inferencer = Inferencer(device, detector, draw_landmarks_on_image,
                            show_original=False, show_fps=True,
                            record_path="tmp/pose_landmarker_test.mp4")
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
