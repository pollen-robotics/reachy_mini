import cv2

from reachy_mini import ReachyMini

rm = ReachyMini(media_backend="gstreamer")

while True:
    frame = rm.media.get_frame()
    if frame is None:
        print("No frame received from the camera.")
        continue
    else:
        print("Frame received:", frame.shape)
        # cv2.imshow("Reachy Mini Camera", frame)
        cv2.imwrite("reachy_mini_camera_frame.jpg", frame)

        # if cv2.waitKey(1) & 0xFF == ord("q"):
        #     break

cv2.destroyAllWindows()
