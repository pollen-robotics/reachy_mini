import time
import numpy as np
from scipy.spatial.transform import Rotation as R

from stewart_little_control import Client
from sixdrepnet import SixDRepNet
import cv2 as cv

model = SixDRepNet(gpu_id=-1)
cap = cv.VideoCapture(0)
# cap = cv.VideoCapture(4)


def main():
    # client = MujocoClient(ip="10.0.0.33")
    client = Client(ip="localhost")

    while True:
        t0 = time.time()
        success, img = cap.read()

        if not success:
            print("Failed to capture image")
            continue

        pitch, yaw, roll = model.predict(img)
        model.draw_axis(img, yaw, pitch, roll)

        roll = roll[0]
        pitch = pitch[0]
        yaw = yaw[0]

        roll = np.clip(roll, -45, 45)
        pitch = np.clip(pitch * 1.5, -45, 45)  # - 20
        yaw = np.clip(yaw, -45, 45)

        # roll = 0
        # pitch = 0
        yaw = 0

        pose = np.eye(4)
        pose[:3, :3] = R.from_euler(
            "xyz", [roll, -pitch, -yaw], degrees=True
        ).as_matrix()
        pose[:3, 3][2] = 0.155

        print(f"roll: {roll}, pitch: {pitch}, yaw: {yaw}")
        # # pose[:3, 3][2] += 0.01 * np.sin(2 * np.pi * 0.5 * time.time())
        client.send_pose(pose, offset_zero=False)
        # time.sleep(0.02)

        cv.imshow("test_window", img)
        cv.waitKey(1)

        t1 = time.time()
        print(f"FPS: {1 / (t1 - t0)}")


if __name__ == "__main__":
    main()
