#!/usr/bin/env python3
"""Acquire calibration images using the camera at maximum resolution.

This script:
1. Sets camera to maximum resolution
2. Captures images when user presses Enter
3. Saves images to ./images/ directory
"""

import argparse
import os
import time
from typing import Any, Optional

import cv2
import numpy.typing as npt

from reachy_mini import ReachyMini  # type: ignore
from reachy_mini.media.camera_constants import CameraResolution


def main() -> None:
    """Acquire calibration images at maximum resolution."""
    parser = argparse.ArgumentParser(
        description="Acquire calibration images at maximum resolution"
    )
    parser.add_argument(
        '--headless',
        action='store_true',
        help='Run in headless mode (no display, use terminal input). For Raspberry Pi/wireless version.'
    )
    parser.add_argument(
        '--save-path',
        type=str,
        default='./images',
        help='Directory to save images (default: ./images)'
    )

    args = parser.parse_args()

    save_path = args.save_path
    os.makedirs(save_path, exist_ok=True)

    print("="*70)
    print("CALIBRATION IMAGE ACQUISITION")
    print("="*70)
    print(f"Save directory: {save_path}")
    print(f"Mode: {'HEADLESS (terminal input)' if args.headless else 'GUI (press Enter to save)'}")
    print("="*70 + "\n")

    # Create window only if not headless
    if not args.headless:
        cv2.namedWindow("Reachy Mini Camera")

    with ReachyMini(media_backend="gstreamer") as reachy_mini:
        if reachy_mini.media.camera is None or reachy_mini.media.camera.camera_specs is None:
            print("ERROR: Could not access camera")
            return

        available_resolutions = reachy_mini.media.camera.camera_specs.available_resolutions

        # Find maximum resolution
        max_resolution: Optional[CameraResolution] = None
        max_resolution_value = 0
        for resolution_enum in available_resolutions:
            res = resolution_enum.value[:2]

            if res[0] * res[1] > max_resolution_value:
                max_resolution_value = res[0] * res[1]
                max_resolution = resolution_enum

        if max_resolution is None:
            print("ERROR: No resolution found")
            return

        res_value = max_resolution.value
        print(f"Using maximum resolution: {max_resolution.name}")
        print(f"  {res_value[0]}x{res_value[1]} @ {res_value[2]}fps\n")

        if reachy_mini.media.camera is not None:
            reachy_mini.media.camera.close()
            reachy_mini.media.camera.set_resolution(max_resolution)
            reachy_mini.media.camera.open()

        time.sleep(2)

        if args.headless:
            print("HEADLESS MODE")
            print("=" * 70)
            print("Instructions:")
            print("1. Position the Charuco board at different angles and distances")
            print("2. Press ENTER to capture an image")
            print("3. Type 'q' and press ENTER to quit")
            print("4. Aim for 20-30 images with varied positions")
            print("=" * 70 + "\n")
        else:
            print("GUI MODE")
            print("=" * 70)
            print("Instructions:")
            print("1. Position the Charuco board at different angles and distances")
            print("2. Press ENTER in the camera window to capture")
            print("3. Press 'q' to quit")
            print("4. Aim for 20-30 images with varied positions")
            print("=" * 70 + "\n")

        try:
            i = 0
            while True:
                if args.headless:
                    # Headless mode: prompt for input
                    user_input = input(f"[{i} images saved] Press ENTER to capture, or type 'q' to quit: ").strip().lower()

                    if user_input == 'q':
                        print("Quitting...")
                        break

                    frame = reachy_mini.media.get_frame()
                    if frame is None:
                        print("ERROR: Failed to grab frame!")
                        continue

                    image_save_path = os.path.join(save_path, f"{i}.png")
                    cv2.imwrite(image_save_path, frame)
                    print(f"✓ Saved: {image_save_path}")
                    i += 1

                else:
                    # GUI mode: show preview
                    frame = reachy_mini.media.get_frame()
                    if frame is None:
                        print("Failed to grab frame.")
                        time.sleep(0.1)
                        continue

                    # Show frame (resized if too large for display)
                    display_frame: npt.NDArray[Any] = frame
                    if frame.shape[0] > 1296:  # If height > 1296, scale down
                        display_frame = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)

                    cv2.imshow("Reachy Mini Camera", display_frame)

                    key = cv2.waitKey(1) & 0xFF
                    if key == 13:  # Enter key
                        image_save_path = os.path.join(save_path, f"{i}.png")
                        cv2.imwrite(image_save_path, frame)
                        print(f"✓ Saved [{i}]: {image_save_path}")
                        i += 1
                    elif key == ord('q'):
                        print("Quitting...")
                        break

                    time.sleep(1.0 / 30)

        except KeyboardInterrupt:
            print("\nInterrupted by user")

        finally:
            if not args.headless:
                cv2.destroyAllWindows()

        print(f"\n{'='*70}")
        print(f"✓ Captured {i} images total")
        print(f"{'='*70}")


if __name__ == "__main__":
    main()
