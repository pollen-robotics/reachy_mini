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
    """Analyze a single image and return marker information."""
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

    # Store marker centers indexed by ID
    marker_centers = {}
    for i, marker_id in enumerate(marker_ids):
        corners = marker_corners[i][0]
        center = corners.mean(axis=0)
        marker_centers[int(marker_id[0])] = center

    return {
        'path': image_path,
        'resolution': (width, height),
        'marker_ids': set(int(id[0]) for id in marker_ids),
        'marker_centers': marker_centers,
        'image': im,
    }


def main():
    aruco_dict, board = build_charuco_board()
    files = sorted(glob("images/CameraResolution.*.png"))

    print("Analyzing images...\n")

    results = {}
    all_marker_ids = None

    for file in files:
        result = analyze_image(file, aruco_dict, board)
        if result:
            name = file.split("/")[1].replace("CameraResolution.", "").replace(".png", "")
            results[name] = result

            print(f"=== {name} ===")
            print(f"Resolution: {result['resolution'][0]}x{result['resolution'][1]}")
            print(f"Detected markers: {len(result['marker_ids'])}")
            print(f"Marker IDs: {sorted(result['marker_ids'])}")
            print()

            if all_marker_ids is None:
                all_marker_ids = result['marker_ids']
            else:
                all_marker_ids = all_marker_ids.intersection(result['marker_ids'])

    print(f"Common markers in ALL images: {sorted(all_marker_ids)}")
    print(f"Number of common markers: {len(all_marker_ids)}\n")

    # Find the reference (largest resolution)
    reference_name = max(results.keys(), key=lambda k: results[k]['resolution'][0] * results[k]['resolution'][1])
    reference = results[reference_name]

    print(f"\n=== Crop/Zoom Analysis (relative to {reference_name}) ===\n")

    # For common markers, calculate distances between pairs
    common_ids = sorted(all_marker_ids)
    if len(common_ids) >= 2:
        # Pick two markers that are far apart for better accuracy
        # Let's use the first and last marker IDs
        id1, id2 = common_ids[0], common_ids[-1]

        print(f"Using markers {id1} and {id2} for distance measurement\n")

        ref_center1 = reference['marker_centers'][id1]
        ref_center2 = reference['marker_centers'][id2]
        ref_distance = np.linalg.norm(ref_center1 - ref_center2)

        for name, result in sorted(results.items(), key=lambda x: x[1]['resolution'][0] * x[1]['resolution'][1], reverse=True):
            center1 = result['marker_centers'][id1]
            center2 = result['marker_centers'][id2]
            distance = np.linalg.norm(center1 - center2)

            # Scale factor: how much larger/smaller do markers appear?
            scale_factor = distance / ref_distance

            # If scale > 1: markers appear larger (more zoomed in / more cropped)
            # If scale < 1: markers appear smaller (less zoomed in / less cropped)

            # FOV factor: inverse of scale
            fov_factor = 1.0 / scale_factor
            crop_percentage = (1.0 - fov_factor) * 100

            print(f"{name}:")
            print(f"  Resolution: {result['resolution'][0]}x{result['resolution'][1]}")
            print(f"  Distance between markers {id1}-{id2}: {distance:.1f} pixels")
            print(f"  Scale factor: {scale_factor:.3f}x (markers appear {scale_factor:.1%} the size of reference)")
            print(f"  Relative FOV: {fov_factor:.2%}")
            print(f"  Crop: {crop_percentage:.1f}% (positive = cropped, negative = wider FOV)")
            print()

    # Also calculate based on image aspect ratio and marker positions
    print("\n=== Alternative Analysis: Markers per image dimension ===\n")

    for name, result in sorted(results.items(), key=lambda x: x[1]['resolution'][0] * x[1]['resolution'][1], reverse=True):
        width, height = result['resolution']

        # Find bounding box of all markers
        all_centers = np.array(list(result['marker_centers'].values()))
        min_x, min_y = all_centers.min(axis=0)
        max_x, max_y = all_centers.max(axis=0)

        board_width_pixels = max_x - min_x
        board_height_pixels = max_y - min_y

        # What fraction of the image does the board occupy?
        width_fraction = board_width_pixels / width
        height_fraction = board_height_pixels / height

        print(f"{name}:")
        print(f"  Board occupies {width_fraction:.1%} of width, {height_fraction:.1%} of height")
        print()


if __name__ == "__main__":
    main()
