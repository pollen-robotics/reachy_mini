import cv2
import numpy as np
from glob import glob
from cv2 import aruco


def build_charuco_board():
    aruco_dict = cv2.aruco.getPredefinedDictionary(aruco.DICT_4X4_1000)
    squares_x = 11          # 11x8 grid
    squares_y = 8
    square_len = 0.02075    # 20.75mm
    marker_len = 0.01558    # 15.58mm
    board = cv2.aruco.CharucoBoard((squares_x, squares_y), square_len, marker_len, aruco_dict)
    return aruco_dict, board


def analyze_image(image_path, aruco_dict, board):
    """Analyze a single image and return charuco corner information."""
    im = cv2.imread(image_path)
    if im is None:
        return None

    height, width = im.shape[:2]

    params = cv2.aruco.DetectorParameters()
    detector = cv2.aruco.ArucoDetector(aruco_dict, params)
    board.setLegacyPattern(True)

    # Detect markers
    marker_corners, marker_ids, rejected = detector.detectMarkers(im)

    if marker_ids is None or len(marker_ids) == 0:
        return None

    # Use marker corners directly
    # Each marker has 4 corners, flatten them all
    all_corners = []
    for marker in marker_corners:
        for corner in marker[0]:
            all_corners.append(corner)

    corners_array = np.array(all_corners)
    min_x = corners_array[:, 0].min()
    max_x = corners_array[:, 0].max()
    min_y = corners_array[:, 1].min()
    max_y = corners_array[:, 1].max()

    board_width = max_x - min_x
    board_height = max_y - min_y

    # Calculate center of the board in image
    center_x = (min_x + max_x) / 2
    center_y = (min_y + max_y) / 2

    return {
        'path': image_path,
        'resolution': (width, height),
        'num_markers': len(marker_ids),
        'board_bbox': (min_x, min_y, max_x, max_y),
        'board_size': (board_width, board_height),
        'board_center': (center_x, center_y),
        'image_center': (width / 2, height / 2),
    }


def main():
    aruco_dict, board = build_charuco_board()
    board.setLegacyPattern(True)
    files = sorted(glob("images/CameraResolution.*.png"))

    print("Analyzing images...\n")

    results = []
    for file in files:
        result = analyze_image(file, aruco_dict, board)
        if result:
            results.append(result)
            name = file.split("/")[1].replace("CameraResolution.", "").replace(".png", "")
            print(f"=== {name} ===")
            print(f"Resolution: {result['resolution'][0]}x{result['resolution'][1]}")
            print(f"Detected markers: {result['num_markers']}")
            print(f"Board size in pixels: {result['board_size'][0]:.1f} x {result['board_size'][1]:.1f}")
            print()

    # Find the reference (largest resolution)
    reference = max(results, key=lambda r: r['resolution'][0] * r['resolution'][1])
    print(f"\n=== Crop Analysis (relative to {reference['path'].split('/')[-1]}) ===\n")

    # Calculate crop factors
    ref_board_width = reference['board_size'][0]
    ref_board_height = reference['board_size'][1]

    for result in results:
        name = result['path'].split("/")[1].replace("CameraResolution.", "").replace(".png", "")

        # Scale factor: how much smaller does the board appear?
        # A smaller board in pixels means we're seeing MORE of the scene (wider FOV, less crop)
        # A larger board in pixels means we're seeing LESS of the scene (narrower FOV, more crop)

        width_scale = result['board_size'][0] / ref_board_width
        height_scale = result['board_size'][1] / ref_board_height

        # FOV factor: inverse of scale
        # If board appears 2x larger, we're seeing 50% of the FOV
        fov_width_factor = 1.0 / width_scale
        fov_height_factor = 1.0 / height_scale

        print(f"{name}:")
        print(f"  Resolution: {result['resolution'][0]}x{result['resolution'][1]}")
        print(f"  Board appears: {width_scale:.2%} width, {height_scale:.2%} height (relative to reference)")
        print(f"  FOV retained: {fov_width_factor:.2%} horizontal, {fov_height_factor:.2%} vertical")
        print(f"  Crop: {(1-fov_width_factor)*100:.1f}% horizontal, {(1-fov_height_factor)*100:.1f}% vertical")
        print()


if __name__ == "__main__":
    main()
