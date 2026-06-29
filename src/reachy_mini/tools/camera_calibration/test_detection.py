"""Quick test to debug Charuco detection."""

import sys
from typing import Tuple

import cv2
from cv2 import aruco


def build_charuco_board() -> Tuple[aruco.Dictionary, aruco.CharucoBoard]:
    """Create and return the Charuco board and dictionary."""
    aruco_dict = cv2.aruco.getPredefinedDictionary(aruco.DICT_4X4_1000)
    squares_x = 11
    squares_y = 8
    square_len = 0.02075
    marker_len = 0.01558
    board = cv2.aruco.CharucoBoard(
        (squares_x, squares_y), square_len, marker_len, aruco_dict
    )
    board.setLegacyPattern(True)
    return aruco_dict, board


# Test with first image
image_path = "images/0.png"
if len(sys.argv) > 1:
    image_path = sys.argv[1]

print(f"Testing detection on: {image_path}")

image = cv2.imread(image_path)
if image is None:
    print(f"ERROR: Could not read image {image_path}")
    exit(1)

print(f"Image size: {image.shape[1]}x{image.shape[0]}")

aruco_dict, board = build_charuco_board()
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Detect markers
params = cv2.aruco.DetectorParameters()
detector = cv2.aruco.ArucoDetector(aruco_dict, params)
marker_corners, marker_ids, rejected = detector.detectMarkers(gray)

print("\nArUco marker detection:")
print(f"  Detected markers: {len(marker_ids) if marker_ids is not None else 0}")
if marker_ids is not None:
    print(f"  Marker IDs: {sorted(marker_ids.flatten().tolist())}")
print(f"  Rejected candidates: {len(rejected)}")

if marker_ids is None or len(marker_ids) == 0:
    print("\nERROR: No ArUco markers detected!")
    print("Possible issues:")
    print("  - Board not in focus or too blurry")
    print("  - Board too small in image")
    print("  - Wrong dictionary (should be DICT_4X4_1000)")
    print("  - Lighting issues")

    # Show image
    cv2.imshow("Image", cv2.resize(image, (0, 0), fx=0.5, fy=0.5))
    print("\nPress any key to close...")
    cv2.waitKey(0)
    exit(1)

# Draw detected markers
vis = image.copy()
cv2.aruco.drawDetectedMarkers(vis, marker_corners, marker_ids)

# Try to detect Charuco corners
print("\nTrying Charuco corner detection...")

try:
    # Using CharucoDetector (correct API for OpenCV 4.x)
    print("Using CharucoDetector.detectBoard()...")
    charuco_detector = cv2.aruco.CharucoDetector(board)
    charuco_corners, charuco_ids, marker_corners_out, marker_ids_out = (
        charuco_detector.detectBoard(gray)
    )

    if charuco_corners is not None and len(charuco_corners) > 0:
        print(f"  SUCCESS! Detected {len(charuco_corners)} Charuco corners")
        print(f"  Corner IDs: {sorted(charuco_ids.flatten().tolist())}")
        # Draw corners
        cv2.aruco.drawDetectedCornersCharuco(vis, charuco_corners, charuco_ids)
    else:
        print("  FAILED: No corners detected")
except Exception as e:
    print(f"  ERROR: {e}")
    import traceback

    traceback.print_exc()

# Show result
cv2.imshow("Detection Result", cv2.resize(vis, (0, 0), fx=0.5, fy=0.5))
print("\nShowing detection result. Press any key to close...")
cv2.waitKey(0)
cv2.destroyAllWindows()
