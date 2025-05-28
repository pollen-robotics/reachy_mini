from hand_tracker import HandTracker
from head_tracker import HeadTracker
import cv2
import numpy as np
import time


class PeekabooDetector:
    def __init__(self):
        self.hand_tracker = HandTracker(nb_hands=2)
        self.head_tracker = HeadTracker()

        self.last_left_eye = 0
        self.last_right_eye = 0
        self.last_time_eyes_update = time.time()
        self.eyes_hidden = False
        self.last_peekaboo = time.time()

    def run(self, img):
        left_eye, right_eye = self.head_tracker.get_eyes(img)
        palm_centers = self.hand_tracker.get_hands_positions(img)
        if left_eye is not None and right_eye is not None:
            self.last_left_eye = left_eye.copy()
            self.last_right_eye = right_eye.copy()
            self.last_time_eyes_update = time.time()
            _left_eye = (left_eye.copy() + 1) / 2  # [0, 1]
            _right_eye = (right_eye.copy() + 1) / 2  # [0, 1]

            h, w, _ = img.shape
            cv2.circle(
                img,
                (int(_left_eye[0] * w), int(_left_eye[1] * h)),
                radius=5,
                color=(0, 0, 255),
                thickness=-1,
            )
            cv2.circle(
                img,
                (int(_right_eye[0] * w), int(_right_eye[1] * h)),
                radius=5,
                color=(0, 0, 255),
                thickness=-1,
            )
        eyes_ok = False
        if time.time() - self.last_time_eyes_update < 1.0:
            eyes_ok = True

        palms_ok = False
        if palm_centers is not None:
            if len(palm_centers) == 2:
                palms_ok = True
            for palm_center in palm_centers:
                h, w, _ = img.shape
                # palm_center : [-1, 1]
                draw_palm = [
                    (palm_center[0] + 1) / 2,
                    (palm_center[1] + 1) / 2,
                ]  # [0, 1]
                cv2.circle(
                    img,
                    (int(w - draw_palm[0] * w), int(draw_palm[1] * h)),
                    radius=5,
                    color=(0, 0, 255),
                    thickness=-1,
                )

        if eyes_ok and palms_ok:
            left_hand, right_hand = palm_centers

            left_dist = np.linalg.norm(
                np.array(left_hand) - np.array(self.last_left_eye)
            )
            right_dist = np.linalg.norm(
                np.array(right_hand) - np.array(self.last_right_eye)
            )

            if left_dist < 0.8 and right_dist < 0.8:
                self.eyes_hidden = True
            elif self.eyes_hidden and time.time() - self.last_peekaboo > 1.0:
                print("PEEKABOO")
                self.eyes_hidden = False
                self.last_peekaboo = time.time()
                return True
        return False

if __name__ == "__main__":
    cap = cv2.VideoCapture(0)
    peekaboo = PeekabooDetector()
    while True:
        success, img = cap.read()
        if not success:
            break

        peekaboo.run(img)
        time.sleep(0.01)
        # cv2.imshow("test_window", img)
        # cv2.waitKey(1)
