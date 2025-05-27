import cv2
import numpy as np
from scipy.spatial.transform import Rotation as R
import mediapipe as mp
from stewart_little_control import Client

cap = cv2.VideoCapture(4)
# cap = cv2.VideoCapture(0)
# while True:

#     success, img = cap.read()

#     cv2.imshow("test_window", img)
#     cv2.waitKey(1)


face_mesh = mp.solutions.face_mesh.FaceMesh(
    min_detection_confidence=0.05,
    min_tracking_confidence=0.5,
    max_num_faces=1,
)


def get_eye_center(face_landmarks):
    left_eye = np.array((face_landmarks.landmark[33].x, face_landmarks.landmark[33].y))
    left_eye = left_eye * 2 - 1

    right_eye = np.array(
        (face_landmarks.landmark[263].x, face_landmarks.landmark[263].y)
    )
    right_eye = right_eye * 2 - 1
    eye_center = np.mean([left_eye, right_eye], axis=0)
    return eye_center


def get_roll(face_landmarks):
    left_eye = np.array((face_landmarks.landmark[33].x, face_landmarks.landmark[33].y))
    left_eye = left_eye * 2 - 1

    right_eye = np.array(
        (face_landmarks.landmark[263].x, face_landmarks.landmark[263].y)
    )
    right_eye = right_eye * 2 - 1
    roll = np.arctan2(right_eye[1] - left_eye[1], right_eye[0] - left_eye[0])

    return roll


client = Client()
pose = np.eye(4)
pose[:3, 3][2] = 0.177  # Set the height of the head
euler_rot = np.array([0.0, 0.0, 0.0])
kp = 0.3
while True:
    success, img = cap.read()

    results = face_mesh.process(img)
    if results.multi_face_landmarks:
        face_landmarks = results.multi_face_landmarks[0]

        # mp.solutions.drawing_utils.draw_landmarks(
        #     image=img,
        #     landmark_list=face_landmarks,
        #     connections=mp.solutions.face_mesh.FACEMESH_CONTOURS,
        #     landmark_drawing_spec=None,
        #     connection_drawing_spec=mp.solutions.drawing_styles.get_default_face_mesh_contours_style(),
        # )

        eye_center = get_eye_center(face_landmarks)  # [-1, 1] [-1, 1]
        _eye_center = (eye_center.copy() + 1) / 2  # [0, 1]
        h, w, _ = img.shape
        cv2.circle(
            img,
            (int(_eye_center[0] * w), int(_eye_center[1] * h)),
            radius=5,
            color=(0, 0, 255),
            thickness=-1,
        )
        _target = [0.5, 0.5]
        cv2.circle(
            img,
            (int(_target[0] * w), int(_target[1] * h)),
            radius=5,
            color=(255, 0, 0),
            thickness=-1,
        )

        cv2.line(
            img,
            (int(_eye_center[0] * w), int(_eye_center[1] * h)),
            (int(_target[0] * w), int(_target[1] * h)),
            color=(0, 255, 0),
            thickness=2,
        )

        roll = get_roll(face_landmarks)
        cv2.line(
            img,
            (int(_eye_center[0] * w), int(_eye_center[1] * h)),
            (
                int(_eye_center[0] * w + 100 * np.cos(roll)),
                int(_eye_center[1] * h + 100 * np.sin(roll)),
            ),
            color=(255, 255, 0),
            thickness=2,
        )

        target = [0, 0]
        error = np.array(target) - eye_center  # [-1, 1] [-1, 1]
        # roll = 0
        euler_rot += np.array([kp*roll*0.1, -kp * 0.1 * error[1], kp * error[0]])
        # euler_rot += np.array([0.0, 0, kp * error[0]])

        rot_mat = R.from_euler("xyz", euler_rot, degrees=False).as_matrix()
        pose[:3, :3] = rot_mat
        pose[:3, 3][2] = (
            error[1] * 0.03 + 0.177
        )  # Adjust height based on vertical error


        client.send_pose(pose)

    cv2.imshow("test_window", img)
    cv2.waitKey(int((1/50)*1000))
