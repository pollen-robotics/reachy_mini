from reachy_mini import ReachyMini
import cv2
import numpy as np
import time

cap = cv2.VideoCapture(4)


def detect_motion(
    img1,
    img2,
    *,
    gaussian_kernel: tuple[int, int] = (5, 5),
    threshold_value: int = 200,
    min_blob_area: int = 100,
):
    blur1 = cv2.GaussianBlur(img1, gaussian_kernel, 0)
    blur2 = cv2.GaussianBlur(img2, gaussian_kernel, 0)
    gray1 = cv2.cvtColor(blur1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(blur2, cv2.COLOR_BGR2GRAY)

    diff = cv2.absdiff(gray1, gray2)
    diff = cv2.normalize(diff, None, 0, 255, cv2.NORM_MINMAX)

    _, mask = cv2.threshold(diff, threshold_value, 255, cv2.THRESH_BINARY)
    kernel = np.ones((3, 3), np.uint8)
    motion_mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=2)
    motion_mask = cv2.dilate(motion_mask, kernel, iterations=1)

    contours, _ = cv2.findContours(
        motion_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    bboxes = []
    overlay = img2.copy()
    for cnt in contours:
        if cv2.contourArea(cnt) < min_blob_area:
            continue
        x, y, w, h = cv2.boundingRect(cnt)
        bboxes.append((x, y, w, h))
        cv2.rectangle(overlay, (x, y), (x + w, y + h), (0, 0, 255), 2)

    return overlay, motion_mask, bboxes


def im_diff(prev, current):
    prev_gray = cv2.cvtColor(prev, cv2.COLOR_BGR2GRAY)
    current_gray = cv2.cvtColor(current, cv2.COLOR_BGR2GRAY)

    prev_gray = prev_gray / 255
    current_gray = current_gray / 255

    diff = np.bitwise_not
    return diff


time_interval = 0.05
prev_frame = None
last_move = time.time()
with ReachyMini() as reachy_mini:
    while True:
        success, frame = cap.read()
        if not success:
            continue

        if prev_frame is not None:
            overlay, mask, boxes = detect_motion(prev_frame, frame)

            cv2.imshow("motion_overlay", overlay)
            cv2.imshow("motion_mask", mask)

            if len(boxes) > 0:
                target_box = boxes[0]
                u = int(np.mean([target_box[0], target_box[0] + target_box[2]]))
                v = int(np.mean([target_box[1], target_box[1] + target_box[3]]))
                cv2.circle(
                    frame, center=(u, v), radius=50, color=(0, 255, 0), thickness=2
                )
                if time.time() - last_move > 2.0:
                    reachy_mini.im_look_at(u, v, duration=0.5)
                    last_move = time.time()
                    prev_frame = None

        cv2.imshow("Frame", frame)
        cv2.waitKey(int(time_interval * 1000))

        if time.time() - last_move < 2.0:
            prev_frame = None

        prev_frame = frame.copy()
