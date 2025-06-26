import cv2 as cv

from cv2_enumerate_cameras import enumerate_cameras, supported_backends


def find_camera_opencv_id(vid='0C45', pid='636D'):
    cameras = enumerate_cameras(cv.CAP_V4L2)

    vid = int(vid, base=16)
    pid = int(pid, base=16)

    for cam_info in cameras:
        if cam_info.pid == pid and cam_info.vid == vid:
            return cam_info.index


def main():
    cap = cv.VideoCapture(find_camera_opencv_id())

    while True:
        b, img = cap.read()
        if b:
            cv.imshow('cam', img)
            cv.waitKey(33)


if __name__ == '__main__':
    main()
