#!/usr/bin/env python3
"""Visualize undistorted camera feed in real-time.

This script:
1. Lets you choose a camera resolution mode
2. Loads the corresponding calibration file
3. Shows live feed with original and undistorted views
4. Press 's' to toggle between split view and full undistorted view
5. Press 'q' to quit
"""

import argparse
import os
import time
from typing import Any, List, Optional, Tuple

import cv2
import numpy as np
import numpy.typing as npt
import yaml

from reachy_mini import ReachyMini  # type: ignore
from reachy_mini.media.camera_constants import CameraResolution


def load_calibration(
    yaml_file: str,
) -> Tuple[npt.NDArray[np.float64], npt.NDArray[np.float64], Tuple[int, int]]:
    """Load calibration parameters from YAML file."""
    with open(yaml_file, "r") as f:
        calib = yaml.safe_load(f)

    # Reconstruct camera matrix
    K = np.array(calib["camera_matrix"]["data"], dtype=np.float64).reshape(3, 3)

    # Get distortion coefficients
    dist = np.array(calib["distortion_coefficients"]["data"], dtype=np.float64)

    # Get image size
    width: int = calib["image_size"]["width"]
    height: int = calib["image_size"]["height"]

    return K, dist, (width, height)


def get_resolution_enum_from_name(
    name: str, available_resolutions: List[CameraResolution]
) -> Optional[CameraResolution]:
    """Get CameraResolution enum from resolution mode name."""
    for res_enum in available_resolutions:
        if res_enum.name == name:
            return res_enum
    return None


def main() -> None:
    """Run the undistortion visualization."""
    parser = argparse.ArgumentParser(description="Visualize undistorted camera feed")
    parser.add_argument(
        "--resolution",
        type=str,
        choices=[
            "R3840x2592at30fps",
            "R3840x2160at30fps",
            "R3264x2448at30fps",
            "R1920x1080at60fps",
        ],
        default=None,
        help="Camera resolution mode to use",
    )
    parser.add_argument(
        "--calibration",
        type=str,
        default=None,
        help="Path to calibration YAML file (auto-detected if not specified)",
    )

    args = parser.parse_args()

    print("=" * 70)
    print("CAMERA UNDISTORTION VISUALIZATION")
    print("=" * 70)

    # Create OpenCV window first (before ReachyMini context)
    window_name = "Camera Undistortion"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    # Connect to robot
    print("\nConnecting to Reachy Mini...")
    reachy_mini = ReachyMini(media_backend="gstreamer")

    if reachy_mini.media.camera is None or reachy_mini.media.camera.camera_specs is None:
        print("ERROR: Could not access camera")
        return

    available_resolutions = reachy_mini.media.camera.camera_specs.available_resolutions

    # Select resolution
    if args.resolution:
        resolution_name = args.resolution
        resolution_enum = get_resolution_enum_from_name(
            resolution_name, available_resolutions
        )
        if resolution_enum is None:
            print(f"ERROR: Resolution {resolution_name} not available")
            return
    else:
        # Let user choose
        print("\nAvailable resolutions:")
        res_list = []
        for i, res in enumerate(available_resolutions):
            res_value = res.value
            print(
                f"  {i + 1}. {res.name} - {res_value[0]}x{res_value[1]} @ {res_value[2]}fps"
            )
            res_list.append(res)

        choice = input("\nSelect resolution (1-{}): ".format(len(res_list)))
        try:
            idx = int(choice) - 1
            resolution_enum = res_list[idx]
            resolution_name = resolution_enum.name
        except (ValueError, IndexError):
            print("Invalid choice")
            return

    print(f"\nUsing resolution: {resolution_name}")

    # Determine calibration file
    if args.calibration:
        calib_file = args.calibration
    else:
        calib_file = f"calibration_{resolution_name}.yaml"

    if not os.path.exists(calib_file):
        print(f"ERROR: Calibration file not found: {calib_file}")
        print("Run calibrate.py and scale_calibration.py first")
        return

    print(f"Loading calibration from: {calib_file}")
    K, dist, (calib_width, calib_height) = load_calibration(calib_file)

    print("\nCalibration parameters:")
    print(f"  Resolution: {calib_width}x{calib_height}")
    print(f"  fx={K[0, 0]:.2f}, fy={K[1, 1]:.2f}")
    print(f"  cx={K[0, 2]:.2f}, cy={K[1, 2]:.2f}")
    print(f"  Distortion coefficients: {len(dist)}")

    # Set camera resolution
    print(f"\nSetting camera to {resolution_name}...")
    if reachy_mini.media.camera is not None:
        reachy_mini.media.camera.close()
        reachy_mini.media.camera.set_resolution(resolution_enum)
        reachy_mini.media.camera.open()

    time.sleep(2)

    # Precompute undistortion maps for better performance
    print("Computing undistortion maps...")
    h, w = calib_height, calib_width
    # alpha=0 removes black borders by cropping to valid pixels only
    new_camera_matrix, roi = cv2.getOptimalNewCameraMatrix(K, dist, (w, h), 0, (w, h))
    mapx, mapy = cv2.initUndistortRectifyMap(
        K, dist, None, new_camera_matrix, (w, h), cv2.CV_32FC1
    )

    # Extract ROI for cropping black borders
    x, y, roi_w, roi_h = roi
    print(f"Valid image region (ROI): {roi_w}x{roi_h} (cropped from {w}x{h})")

    print("\n" + "=" * 70)
    print("CONTROLS:")
    print("  's' - Toggle between split view and full undistorted view")
    print("  'q' - Quit")
    print("=" * 70 + "\n")

    split_view = True

    try:
        while True:
            frame = reachy_mini.media.get_frame()
            if frame is None:
                print("Failed to grab frame")
                time.sleep(0.1)
                continue

            # Undistort using precomputed maps (faster)
            undistorted = cv2.remap(frame, mapx, mapy, cv2.INTER_LINEAR)

            # Crop to valid region (removes black borders)
            undistorted = undistorted[y : y + roi_h, x : x + roi_w]

            # Create visualization
            if split_view:
                # Side-by-side view - resize undistorted to match original height for comparison
                orig_h, orig_w = frame.shape[:2]
                undist_h, undist_w = undistorted.shape[:2]

                # Resize undistorted to match original height
                scale = orig_h / undist_h
                new_undist_w = int(undist_w * scale)
                undistorted_resized = cv2.resize(undistorted, (new_undist_w, orig_h))

                combined: npt.NDArray[Any] = np.zeros((orig_h, orig_w + new_undist_w, 3), dtype=np.uint8)
                combined[:, :orig_w] = frame
                combined[:, orig_w : orig_w + new_undist_w] = undistorted_resized

                # Add labels
                cv2.putText(
                    combined,
                    "Original",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 0, 255),
                    2,
                )
                cv2.putText(
                    combined,
                    "Undistorted (no borders)",
                    (orig_w + 10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 255, 0),
                    2,
                )

                display = combined
            else:
                # Full undistorted view
                display = undistorted
                cv2.putText(
                    display,
                    "Undistorted - No borders (press 's' for split view)",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 0),
                    2,
                )

            # Display
            cv2.imshow(window_name, display)

            # Handle keys
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            elif key == ord("s"):
                split_view = not split_view
                print(f"View mode: {'split' if split_view else 'full undistorted'}")

            time.sleep(1.0 / 30)

    except KeyboardInterrupt:
        print("\nInterrupted by user")

    finally:
        # Close camera first, then destroy windows
        try:
            if reachy_mini.media.camera is not None:
                reachy_mini.media.camera.close()
                print("\nClosed camera")
        except Exception:
            pass
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
