import cv2
import logging

from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision
import time

logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


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


def capture(dev):
    logger.info("loading media pipe pose landmarker")

    base_options = mp_python.BaseOptions(
        model_asset_path='tmp/pose_landmarker.task',
        # delegate=mp_python.BaseOptions.Delegate.GPU
    )
    options = mp_vision.PoseLandmarkerOptions(
        base_options=base_options,
        output_segmentation_masks=True)
    detector = mp_vision.PoseLandmarker.create_from_options(options)

    cap = cv2.VideoCapture(dev)

    if not cap.isOpened():
        logger.error(f"Failed to open device {dev}")
        return None
    ret = True

    im_width, im_height = (cap.get(cv2.CAP_PROP_FRAME_WIDTH), cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    time_start = time.perf_counter()
    while ret:
        ret, frame = cap.read()
        if not ret:
            logger.error(f"Failed to read from device {dev}")
            return None
        frame_np = np.array(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        image = mp.Image(
            image_format=mp.ImageFormat.SRGB, data=frame_np,
            # channels=3, width=im_width, height=im_height
        )

        # STEP 4: Detect pose landmarks from the input image.
        detection_result = detector.detect(image)

        # STEP 5: Process the detection result. In this case, visualize it.
        annotated_image = draw_landmarks_on_image(image.numpy_view(), detection_result)
        cv2.imshow("labeled", cv2.cvtColor(annotated_image, cv2.COLOR_RGB2BGR))

        fps = 1.0 / (time.perf_counter() - time_start)
        time_start = time.perf_counter()
        logger.info(f"fps: {fps:.2f}")

        if not ret:
            logger.error(f"Failed to read from device {dev}")
            return None
        # cv2.imshow(f"Device {dev}", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break


if __name__ == "__main__":
    try:
        capture(1)
    except KeyboardInterrupt as e:
        logger.warning("exit on KeyboardInterrupt")
    except Exception as e:
        logger.error(f"exit on {e}")
    finally:

        cv2.destroyAllWindows()
