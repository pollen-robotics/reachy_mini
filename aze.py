from reachy_mini import ReachyMini
import cv2
import numpy as np

# capture = cv2.VideoCapture(1)
# while True:
#     ret, frame = capture.read()
#     cv2.imshow('test', frame)
#     cv2.waitKey(1)


rm = ReachyMini(media_backend="no_media")
rm.goto_target(np.eye(4), antennas=[1, 0])