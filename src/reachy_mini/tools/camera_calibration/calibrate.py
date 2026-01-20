#!/usr/bin/env python3
"""Camera calibration script using Charuco board.

This script:
1. Reads calibration images from ./images/ directory
2. Detects Charuco board corners in each image
3. Performs camera calibration
4. Saves calibration parameters to calibration.yaml

Usage:
    python calibrate.py [--images-dir ./images] [--output calibration.yaml]
"""

import argparse
import os
from glob import glob
from typing import Any, List, Optional, Tuple

import cv2
import numpy as np
import numpy.typing as npt
import yaml
from cv2 import aruco


def build_charuco_board() -> Tuple[aruco.Dictionary, aruco.CharucoBoard]:
    """Build the Charuco board definition matching the physical board."""
    aruco_dict = cv2.aruco.getPredefinedDictionary(aruco.DICT_4X4_1000)

    # Board parameters (must match physical board!)
    squares_x = 11  # number of chessboard squares in X
    squares_y = 8  # number of chessboard squares in Y
    square_len = 0.02075  # meters (20.75mm)
    marker_len = 0.01558  # meters (15.58mm)

    board = cv2.aruco.CharucoBoard(
        (squares_x, squares_y), square_len, marker_len, aruco_dict
    )
    return aruco_dict, board


def detect_charuco_corners(
    image: npt.NDArray[Any],
    aruco_dict: aruco.Dictionary,
    board: aruco.CharucoBoard,
    min_markers: int = 4,
) -> Tuple[Optional[npt.NDArray[Any]], Optional[npt.NDArray[Any]], bool]:
    """Detect Charuco corners in an image.

    Returns:
        charuco_corners: Detected corner positions
        charuco_ids: IDs of detected corners
        success: Whether detection was successful

    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Detect ArUco markers
    params = cv2.aruco.DetectorParameters()
    detector = cv2.aruco.ArucoDetector(aruco_dict, params)
    board.setLegacyPattern(True)
    marker_corners, marker_ids, rejected = detector.detectMarkers(gray)

    if marker_ids is None or len(marker_ids) < min_markers:
        return None, None, False

    # Interpolate Charuco corners using the charuco detector
    charuco_detector = cv2.aruco.CharucoDetector(board)
    charuco_corners, charuco_ids, marker_corners_out, marker_ids_out = (
        charuco_detector.detectBoard(gray)
    )

    if charuco_corners is None or len(charuco_corners) < min_markers:
        return None, None, False

    return charuco_corners, charuco_ids, True


def calibrate_camera(
    images_dir: str, min_markers: int = 20, visualize: bool = False
) -> Tuple[
    npt.NDArray[np.float64],
    npt.NDArray[np.float64],
    List[npt.NDArray[np.float64]],
    List[npt.NDArray[np.float64]],
    Tuple[int, int],
    float,
    List[str],
]:
    """Perform camera calibration using images from the specified directory.

    Args:
        images_dir: Directory containing calibration images
        min_markers: Minimum number of detected corners required per image
        visualize: Whether to display images with detected corners

    Returns:
        camera_matrix: 3x3 camera intrinsic matrix
        dist_coeffs: Distortion coefficients
        rvecs: Rotation vectors for each image
        tvecs: Translation vectors for each image
        image_size: (width, height) of calibration images
        rms_error: RMS reprojection error

    """
    aruco_dict, board = build_charuco_board()

    # Find all PNG images
    image_files = sorted(glob(os.path.join(images_dir, "*.png")))

    if not image_files:
        raise ValueError(f"No PNG images found in {images_dir}")

    print(f"Found {len(image_files)} images in {images_dir}")

    # Storage for calibration data
    all_charuco_corners: List[npt.NDArray[Any]] = []
    all_charuco_ids: List[npt.NDArray[Any]] = []
    image_size: Optional[Tuple[int, int]] = None
    successful_images: List[str] = []

    # Process each image
    for i, image_file in enumerate(image_files):
        print(
            f"\nProcessing {os.path.basename(image_file)} ({i + 1}/{len(image_files)})...",
            end=" ",
        )

        image = cv2.imread(image_file)
        if image is None:
            print("Failed to read!")
            continue

        if image_size is None:
            image_size = (image.shape[1], image.shape[0])

        # Detect corners
        charuco_corners, charuco_ids, success = detect_charuco_corners(
            image, aruco_dict, board, min_markers
        )

        if not success or charuco_corners is None or charuco_ids is None:
            print("Detection failed!")
            continue

        print(f"Detected {len(charuco_corners)} corners")

        all_charuco_corners.append(charuco_corners)
        all_charuco_ids.append(charuco_ids)
        successful_images.append(image_file)

        # Visualize if requested
        if visualize:
            vis_image = image.copy()
            cv2.aruco.drawDetectedCornersCharuco(
                vis_image, charuco_corners, charuco_ids
            )
            cv2.imshow(
                "Detected Corners", cv2.resize(vis_image, (0, 0), fx=0.5, fy=0.5)
            )
            cv2.waitKey(500)

    if visualize:
        cv2.destroyAllWindows()

    print(
        f"\n\nSuccessfully processed {len(all_charuco_corners)}/{len(image_files)} images"
    )

    if len(all_charuco_corners) < 3:
        raise ValueError(
            f"Need at least 3 successful images for calibration, got {len(all_charuco_corners)}"
        )

    # Perform calibration
    print("\nPerforming camera calibration...")

    # Ensure image_size is set
    if image_size is None:
        raise ValueError("Failed to determine image size from calibration images")

    calibration_flags = (
        cv2.CALIB_RATIONAL_MODEL  # Use rational distortion model (k4, k5, k6)
        | cv2.CALIB_THIN_PRISM_MODEL  # Use thin prism distortion (s1, s2, s3, s4)
    )

    # Use cv2.calibrateCamera with object points from the board
    # First, get 3D object points for each detected corner
    all_obj_points: List[npt.NDArray[Any]] = []
    all_img_points: List[npt.NDArray[Any]] = []

    for corners, ids in zip(all_charuco_corners, all_charuco_ids):
        # Get 3D points from board
        chessboard_corners = board.getChessboardCorners()
        obj_pts = np.array([chessboard_corners[i] for i in ids.flatten().tolist()], dtype=np.float32)
        all_obj_points.append(obj_pts)
        all_img_points.append(corners)

    # cv2.calibrateCamera returns (rms_error, camera_matrix, dist_coeffs, rvecs, tvecs)
    calib_result = cv2.calibrateCamera(  # type: ignore[call-overload]
        objectPoints=all_obj_points,
        imagePoints=all_img_points,
        imageSize=image_size,
        cameraMatrix=None,
        distCoeffs=None,
        flags=calibration_flags,
    )
    rms_error = calib_result[0]
    camera_matrix = calib_result[1]
    dist_coeffs = calib_result[2]
    rvecs = calib_result[3]
    tvecs = calib_result[4]

    print(f"\n{'=' * 60}")
    print("CALIBRATION RESULTS")
    print(f"{'=' * 60}")
    print(f"RMS Reprojection Error: {rms_error:.4f} pixels")
    print(f"Image size: {image_size[0]}x{image_size[1]}")
    print(f"Images used: {len(all_charuco_corners)}")
    print("\nCamera Matrix (K):")
    print(camera_matrix)
    print("\nDistortion Coefficients:")
    print(f"  k1={dist_coeffs[0, 0]:.6f}, k2={dist_coeffs[0, 1]:.6f}")
    print(f"  p1={dist_coeffs[0, 2]:.6f}, p2={dist_coeffs[0, 3]:.6f}")
    print(f"  k3={dist_coeffs[0, 4]:.6f}", end="")
    if dist_coeffs.shape[1] > 5:
        print(
            f", k4={dist_coeffs[0, 5]:.6f}, k5={dist_coeffs[0, 6]:.6f}, k6={dist_coeffs[0, 7]:.6f}",
            end="",
        )
    if dist_coeffs.shape[1] > 8:
        print(
            f"\n  s1={dist_coeffs[0, 8]:.6f}, s2={dist_coeffs[0, 9]:.6f}, s3={dist_coeffs[0, 10]:.6f}, s4={dist_coeffs[0, 11]:.6f}"
        )
    else:
        print()
    print(f"{'=' * 60}\n")

    return (
        camera_matrix,
        dist_coeffs,
        rvecs,
        tvecs,
        image_size,
        rms_error,
        successful_images,
    )


def save_calibration(
    output_file: str,
    camera_matrix: npt.NDArray[np.float64],
    dist_coeffs: npt.NDArray[np.float64],
    image_size: Tuple[int, int],
    rms_error: float,
    successful_images: List[str],
) -> None:
    """Save calibration parameters to a YAML file."""
    calibration_data = {
        "calibration_date": str(np.datetime64("now")),
        "image_size": {"width": int(image_size[0]), "height": int(image_size[1])},
        "camera_matrix": {
            "rows": 3,
            "cols": 3,
            "data": camera_matrix.flatten().tolist(),
        },
        "distortion_coefficients": {
            "rows": 1,
            "cols": dist_coeffs.shape[1],
            "data": dist_coeffs.flatten().tolist(),
        },
        "distortion_model": "rational_polynomial"
        if dist_coeffs.shape[1] >= 8
        else "plumb_bob",
        "rms_error": float(rms_error),
        "num_images": len(successful_images),
        "image_files": [os.path.basename(f) for f in successful_images],
    }

    # Also save in OpenCV format for easy loading
    calibration_data["fx"] = float(camera_matrix[0, 0])
    calibration_data["fy"] = float(camera_matrix[1, 1])
    calibration_data["cx"] = float(camera_matrix[0, 2])
    calibration_data["cy"] = float(camera_matrix[1, 2])

    with open(output_file, "w") as f:
        yaml.dump(calibration_data, f, default_flow_style=False, sort_keys=False)

    print(f"Calibration saved to {output_file}")


def main() -> None:
    """Run the calibration process based on command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Calibrate camera using Charuco board images"
    )
    parser.add_argument(
        "--images-dir",
        type=str,
        default="./images",
        help="Directory containing calibration images (default: ./images)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="calibration.yaml",
        help="Output calibration file (default: calibration.yaml)",
    )
    parser.add_argument(
        "--min-markers",
        type=int,
        default=20,
        help="Minimum number of detected corners per image (default: 20)",
    )
    parser.add_argument(
        "--visualize",
        action="store_true",
        help="Show detected corners during processing",
    )

    args = parser.parse_args()

    # Verify images directory exists
    if not os.path.isdir(args.images_dir):
        print(f"Error: Images directory '{args.images_dir}' does not exist")
        return

    try:
        # Perform calibration
        (
            camera_matrix,
            dist_coeffs,
            rvecs,
            tvecs,
            image_size,
            rms_error,
            successful_images,
        ) = calibrate_camera(
            args.images_dir, min_markers=args.min_markers, visualize=args.visualize
        )

        # Save results
        save_calibration(
            args.output,
            camera_matrix,
            dist_coeffs,
            image_size,
            rms_error,
            successful_images,
        )

        print("\nCalibration completed successfully!")

    except Exception as e:
        print(f"\nError during calibration: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
