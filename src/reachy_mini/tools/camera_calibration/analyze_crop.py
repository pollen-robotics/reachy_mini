"""Analyze crop and zoom differences between images using ArUco markers."""

from glob import glob
from typing import Any, Dict, Optional, Set, Tuple

import cv2
import numpy as np
import numpy.typing as npt
from cv2 import aruco


def build_charuco_board() -> Tuple[aruco.Dictionary, aruco.CharucoBoard]:
    """Build the Charuco board used for calibration."""
    aruco_dict = cv2.aruco.getPredefinedDictionary(aruco.DICT_4X4_1000)
    squares_x = 11  # 11x8 grid
    squares_y = 8
    square_len = 0.02075  # 20.75mm
    marker_len = 0.01558  # 15.58mm
    board = cv2.aruco.CharucoBoard(
        (squares_x, squares_y), square_len, marker_len, aruco_dict
    )
    return aruco_dict, board


def analyze_image(
    image_path: str, aruco_dict: aruco.Dictionary, board: aruco.CharucoBoard
) -> Optional[Dict[str, Any]]:
    """Analyze a single image and return marker information."""
    im = cv2.imread(image_path)
    if im is None:
        return None

    height, width = im.shape[:2]

    params = cv2.aruco.DetectorParameters()
    detector = cv2.aruco.ArucoDetector(aruco_dict, params)
    board.setLegacyPattern(True)

    marker_corners, marker_ids, rejected = detector.detectMarkers(im)

    if marker_ids is None or len(marker_ids) == 0:
        return None

    # Store marker centers indexed by ID (normalized to [0,1] range)
    marker_centers: Dict[int, npt.NDArray[np.float64]] = {}
    marker_centers_pixels: Dict[int, npt.NDArray[Any]] = {}
    for i, marker_id in enumerate(marker_ids):
        corners = marker_corners[i][0]
        center = corners.mean(axis=0)
        marker_centers_pixels[int(marker_id[0])] = center
        # Normalize to image dimensions
        marker_centers[int(marker_id[0])] = np.array(
            [float(center[0]) / width, float(center[1]) / height]
        )

    marker_ids_set: Set[int] = set(int(mid[0]) for mid in marker_ids)

    return {
        "path": image_path,
        "resolution": (width, height),
        "marker_ids": marker_ids_set,
        "marker_centers": marker_centers,  # normalized
        "marker_centers_pixels": marker_centers_pixels,  # absolute
        "image": im,
    }


def main() -> None:
    """Run the analysis on all images in the 'images' folder."""
    aruco_dict, board = build_charuco_board()
    files = sorted(glob("images/*.png"))

    print("Analyzing images...\n")

    results: Dict[str, Dict[str, Any]] = {}
    all_marker_ids: Optional[Set[int]] = None

    for file in files:
        result = analyze_image(file, aruco_dict, board)
        if result:
            name = (
                file.split("/")[1].replace("CameraResolution.", "").replace(".png", "")
            )
            results[name] = result

            print(f"=== {name} ===")
            print(f"Resolution: {result['resolution'][0]}x{result['resolution'][1]}")
            print(f"Detected markers: {len(result['marker_ids'])}")
            print(f"Marker IDs: {sorted(result['marker_ids'])}")
            print()

            if all_marker_ids is None:
                all_marker_ids = result["marker_ids"]
            else:
                all_marker_ids = all_marker_ids.intersection(result["marker_ids"])

    if all_marker_ids is None:
        print("No marker IDs found in any images")
        return

    print(f"Common markers in ALL images: {sorted(all_marker_ids)}")
    print(f"Number of common markers: {len(all_marker_ids)}\n")

    # Find the reference (largest resolution)
    reference_name = max(
        results.keys(),
        key=lambda k: results[k]["resolution"][0] * results[k]["resolution"][1],
    )
    reference = results[reference_name]

    print(f"\n=== Crop/Zoom Analysis (relative to {reference_name}) ===")
    print("Using NORMALIZED distances (as fraction of image size)\n")

    common_ids = sorted(all_marker_ids)
    if len(common_ids) >= 2:
        # Pick markers far apart
        id1, id2 = common_ids[0], common_ids[-1]

        print(f"Using markers {id1} and {id2} for distance measurement\n")

        # Calculate distance in normalized coordinates
        ref_center1 = reference["marker_centers"][id1]
        ref_center2 = reference["marker_centers"][id2]
        ref_distance = np.linalg.norm(ref_center1 - ref_center2)

        for name, result in sorted(
            results.items(),
            key=lambda x: x[1]["resolution"][0] * x[1]["resolution"][1],
            reverse=True,
        ):
            center1 = result["marker_centers"][id1]
            center2 = result["marker_centers"][id2]
            distance = np.linalg.norm(center1 - center2)

            # Now this is the true scale comparison
            # If distance > ref_distance: markers are farther apart in frame = board takes up more of image = MORE ZOOM/CROP
            # If distance < ref_distance: markers are closer together = board takes up less of image = LESS ZOOM
            scale_factor = distance / ref_distance

            fov_factor = 1.0 / scale_factor

            print(f"{name}:")
            print(f"  Resolution: {result['resolution'][0]}x{result['resolution'][1]}")
            print(
                f"  Normalized distance: {distance:.4f} (reference: {ref_distance:.4f})"
            )
            print(f"  Board occupies {scale_factor:.2%} of reference span")
            if scale_factor > 1:
                print(
                    f"  → Board appears LARGER = MORE ZOOMED IN/CROPPED by {((scale_factor - 1) * 100):.1f}%"
                )
            else:
                print(
                    f"  → Board appears SMALLER = LESS ZOOMED IN by {((1 - scale_factor) * 100):.1f}%"
                )
            print(f"  Effective FOV: {fov_factor:.2%} of reference")
            print()

    # Count markers visible in each
    print("\n=== Marker Visibility Analysis ===\n")

    ref_markers = results[reference_name]["marker_ids"]

    for name, result in sorted(
        results.items(),
        key=lambda x: x[1]["resolution"][0] * x[1]["resolution"][1],
        reverse=True,
    ):
        markers = result["marker_ids"]
        missing = ref_markers - markers
        print(f"{name}:")
        print(f"  Visible markers: {len(markers)}/{len(ref_markers)}")
        print(f"  Missing markers: {sorted(missing) if missing else 'none'}")
        print()


if __name__ == "__main__":
    main()
