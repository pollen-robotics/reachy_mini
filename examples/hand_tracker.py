import mediapipe as mp
import numpy as np
import cv2

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands

# cap = cv2.VideoCapture(4)
# cap = cv2.VideoCapture(0)


class HandTracker:
    def __init__(self, nb_hands=1):
        self.hands = mp_hands.Hands(
            static_image_mode=True, max_num_hands=nb_hands, min_detection_confidence=0.5
        )

    def get_hands_positions(self, img):
        img = cv2.flip(img, 1)

        results = self.hands.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        if results.multi_hand_landmarks is not None:
            # for hand_landmarks in results.multi_hand_landmarks:
            #     mp_drawing.draw_landmarks(
            #         img,
            #         hand_landmarks,
            #         mp_hands.HAND_CONNECTIONS,
            #         mp_drawing_styles.get_default_hand_landmarks_style(),
            #         mp_drawing_styles.get_default_hand_connections_style(),
            #     )
            # palm_base_landmark = results.multi_hand_landmarks[0].landmark[
            #     mp_hands.HandLandmark.WRIST
            # ]
            # # palm_center = [(palm_base_landmark.x-0.5)*2, (palm_base_landmark.y-0.5)*2]
            # middle_finger_base_landmark = results.multi_hand_landmarks[0].landmark[
            #     mp_hands.HandLandmark.MIDDLE_FINGER_MCP
            # ]
            palm_centers = []
            for landmarks in results.multi_hand_landmarks:
                # palm_base_landmark = landmarks.landmark[mp_hands.HandLandmark.WRIST]
                # middle_finger_base_landmark = landmarks.landmark[
                #     mp_hands.HandLandmark.MIDDLE_FINGER_MCP
                # ]
                middle_finger_pip_landmark = landmarks.landmark[
                    mp_hands.HandLandmark.MIDDLE_FINGER_PIP
                ]
                # vec = (
                #     np.array(
                #         [
                #             palm_base_landmark.x - middle_finger_base_landmark.x + -0.1,
                #             palm_base_landmark.y - middle_finger_base_landmark.y ,
                #         ]
                #     )
                #     / 2.8
                # )
                palm_center = np.array(
                    [middle_finger_pip_landmark.x, middle_finger_pip_landmark.y]
                )

                h, w, _ = img.shape

                palm_center = [(palm_center[0] - 0.5) * 2, (palm_center[1] - 0.5) * 2]
                # palm_center = [(palm_landmark.x-0.5)*2, (palm_landmark.y-0.5)*2]
                palm_centers.append(palm_center)

            return palm_centers
        return None


if __name__ == "__main__":
    # cap = cv2.VideoCapture(4)
    cap = cv2.VideoCapture(0)
    hand_tracker = HandTracker(nb_hands=2)
    while True:
        success, img = cap.read()
        if not success:
            break
        palm_centers = hand_tracker.get_hands_positions(img)
        if palm_centers is not None:
            for palm_center in palm_centers:
                print("Palm Center:", palm_center)
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

        cv2.imshow("Hand Tracking", cv2.flip(img, 1))
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    cap.release()
    cv2.destroyAllWindows()
