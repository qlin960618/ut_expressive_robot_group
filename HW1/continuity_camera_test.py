import cv2
import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


def capture(dev):
    cap = cv2.VideoCapture(dev)
    if not cap.isOpened():
        logger.error(f"Failed to open device {dev}")
        return None
    ret = True
    while ret:
        ret, frame = cap.read()
        if not ret:
            logger.error(f"Failed to read from device {dev}")
            return None
        cv2.imshow(f"Device {dev}", frame)
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

