from reachy_mini import Client
from peekaboo import PeekabooDetector
import cv2
import numpy as np
import time

client = Client()
peekaboo = PeekabooDetector()

cap = cv2.VideoCapture(0)
init_pose = np.eye(4)
init_pose[:3, 3][2] = 0.177  # Set the height of the head

peekaboo_pose = init_pose.copy()
peekaboo_pose[:3, 3][2] = 0.15
last_peekaboo = time.time()
peekabooing = False

while True:
    success, img = cap.read()
    if not success:
        break

    if peekaboo.run(img) and not peekabooing:
        peekabooing = True
        last_peekaboo = time.time()

    if peekabooing:
        if time.time() - last_peekaboo > 1.0:
            peekabooing = False

        client.send_pose(peekaboo_pose, offset_zero=False)

    else:
        client.send_pose(init_pose, offset_zero=False)

    # time.sleep(0.01)
    cv2.imshow("test_window", img)
    cv2.waitKey(1)
