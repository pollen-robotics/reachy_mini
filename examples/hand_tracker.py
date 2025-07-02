"""Hand Tracker using MediaPipe to detect hand positions in images."""

try:
    import mediapipe as mp
except ImportError:
    print("mediapipe is not installed. Install it using 'pip install mediapipe'.")
    exit()

import cv2
import numpy as np

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands


class HandTracker:
    """Hand Tracker using MediaPipe Hands to detect hand positions."""

    def __init__(self, nb_hands=1):
        """Initialize the Hand Tracker."""
        self.hands = mp_hands.Hands(
            static_image_mode=True, max_num_hands=nb_hands, min_detection_confidence=0.5
        )

    def get_hands_positions(self, img):
        """Get the positions of the hands in the image."""
        img = cv2.flip(img, 1)

        results = self.hands.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        if results.multi_hand_landmarks is not None:
            palm_centers = []
            for landmarks in results.multi_hand_landmarks:
                middle_finger_pip_landmark = landmarks.landmark[
                    mp_hands.HandLandmark.MIDDLE_FINGER_PIP
                ]
                palm_center = np.array(
                    [middle_finger_pip_landmark.x, middle_finger_pip_landmark.y]
                )

                h, w, _ = img.shape

                palm_center = [(palm_center[0] - 0.5) * 2, (palm_center[1] - 0.5) * 2]
                palm_centers.append(palm_center)

            return palm_centers
        return None
