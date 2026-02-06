#!/usr/bin/env python3
"""Acquire images at different resolutions for crop analysis.

This script cycles through all available camera resolutions and saves
one image per resolution for analyzing how much each mode crops.
"""

import argparse
import os
import time
from typing import Any

import cv2
import numpy.typing as npt

from reachy_mini import ReachyMini  # type: ignore


def main() -> None:
    """Acquire images at different resolutions for crop analysis."""
    parser = argparse.ArgumentParser(
        description="Acquire images at different resolutions for crop analysis"
    )
    parser.add_argument(
        "--headless",
        action="store_true",
        help="Run in headless mode (no display, use terminal input). For Raspberry Pi/wireless version.",
    )
    parser.add_argument(
        "--save-path",
        type=str,
        default="./images",
        help="Directory to save images (default: ./images)",
    )

    args = parser.parse_args()

    save_path = args.save_path
    os.makedirs(save_path, exist_ok=True)

    print("=" * 70)
    print("CROP ANALYSIS IMAGE ACQUISITION")
    print("=" * 70)
    print(f"Save directory: {save_path}")
    print(
        f"Mode: {'HEADLESS (terminal input)' if args.headless else 'GUI (press Enter to save)'}"
    )
    print("=" * 70 + "\n")

    # Create window only if not headless
    if not args.headless:
        cv2.namedWindow("Reachy Mini Camera")

    with ReachyMini(media_backend="gstreamer") as reachy_mini:
        if reachy_mini.media.camera is None or reachy_mini.media.camera.camera_specs is None:
            print("ERROR: Could not access camera")
            return

        available_resolutions = (
            reachy_mini.media.camera.camera_specs.available_resolutions
        )

        # Sort by resolution size (largest first)
        sorted_resolutions = sorted(
            available_resolutions, key=lambda r: r.value[0] * r.value[1], reverse=True
        )

        print(f"Found {len(sorted_resolutions)} resolutions:")
        for i, res in enumerate(sorted_resolutions):
            res_value = res.value
            print(
                f"  {i + 1}. {res.name} - {res_value[0]}x{res_value[1]} @ {res_value[2]}fps"
            )
        print()

        try:
            for i, current_res in enumerate(sorted_resolutions):
                res_value = current_res.value
                print(
                    f"\n[{i + 1}/{len(sorted_resolutions)}] Setting resolution to {current_res.name}"
                )
                print(
                    f"  Resolution: {res_value[0]}x{res_value[1]} @ {res_value[2]}fps"
                )

                if reachy_mini.media.camera is not None:
                    reachy_mini.media.camera.close()
                    time.sleep(1)
                    reachy_mini.media.camera.set_resolution(current_res)
                    time.sleep(1)
                    reachy_mini.media.camera.open()
                    time.sleep(2)  # Wait for camera to stabilize

                image_save_path = os.path.join(save_path, f"{current_res.name}.png")

                if args.headless:
                    # Headless mode: use terminal input
                    print("\nPosition the camera and Charuco board.")
                    input("Press ENTER when ready to capture...")

                    frame = reachy_mini.media.get_frame()
                    if frame is None:
                        print("ERROR: Failed to grab frame!")
                        continue

                    cv2.imwrite(image_save_path, frame)
                    print(f"✓ Saved: {image_save_path}")

                else:
                    # GUI mode: show preview and wait for Enter key
                    print(
                        "Press ENTER in the camera window to capture, or 'q' to quit..."
                    )

                    saved = False
                    while not saved:
                        frame = reachy_mini.media.get_frame()
                        if frame is None:
                            print("Failed to grab frame.")
                            time.sleep(0.1)
                            continue

                        # Show frame (resized if too large)
                        display_frame: npt.NDArray[Any] = frame
                        if frame.shape[0] > 1296:  # If height > 1296, scale down
                            display_frame = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)

                        cv2.imshow("Reachy Mini Camera", display_frame)

                        key = cv2.waitKey(1) & 0xFF
                        if key == 13:  # Enter key
                            cv2.imwrite(image_save_path, frame)
                            print(f"✓ Saved: {image_save_path}")
                            saved = True
                        elif key == ord("q"):
                            print("Quit requested")
                            return

                        time.sleep(1.0 / 30)

            print("\n" + "=" * 70)
            print(f"✓ All done! Captured {len(sorted_resolutions)} images")
            print("=" * 70)

        except KeyboardInterrupt:
            print("\nInterrupted by user")

        finally:
            if not args.headless:
                cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
