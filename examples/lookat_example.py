from reachy_mini import ReachyMini
import cv2
import numpy as np

from reachy_mini.io.cam_utils import find_camera

cap = find_camera()

def click(event, x, y, flags, param):
    global click_x, click_y, just_clicked

    if event == cv2.EVENT_LBUTTONDOWN:
        just_clicked = True
        click_x, click_y = x, y


click_x, click_y = 0, 0
just_clicked = False
frame = np.zeros((1280, 720, 3))
cv2.namedWindow("Frame")
cv2.setMouseCallback("Frame", click)

print("Click on the image to make ReachyMini look at that point.")
with ReachyMini() as reachy_mini:
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame.")
            continue

        cv2.imshow("Frame", frame)
        cv2.waitKey(1)

        if just_clicked:
            reachy_mini.look_at_image(click_x, click_y, duration=0.3)
            just_clicked = False
