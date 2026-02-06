#!/usr/bin/env python3
"""Scale camera calibration parameters for different resolution modes.

When the camera uses a cropped sensor region or different resolution,
the intrinsic parameters need to be adjusted accordingly.

This script takes calibration parameters from the full resolution (3840x2592)
and computes adjusted parameters for other resolution modes.
"""

import argparse
from typing import Any, Dict, List, Tuple

import numpy as np
import numpy.typing as npt
import yaml

# Crop analysis results from analyze_crop_v3.py
RESOLUTION_MODES = {
    "R3840x2592at30fps": {
        "resolution": (3840, 2592),
        "crop_scale": 1.0,  # Reference - no crop
        "description": "Full resolution (reference)",
    },
    "R3840x2160at30fps": {
        "resolution": (3840, 2160),
        "crop_scale": 1.109,  # Board appears 11% larger -> 11% digital zoom
        "description": "4K UHD - 11% crop (vertical)",
    },
    "R3264x2448at30fps": {
        "resolution": (3264, 2448),
        "crop_scale": 1.115,  # Board appears 11.5% larger
        "description": "3MP - 11.5% crop (both axes)",
    },
    "R1920x1080at60fps": {
        "resolution": (1920, 1080),
        "crop_scale": 1.115,  # Board appears 11.5% larger
        "description": "Full HD 60fps - 11.5% crop (vertical)",
    },
}


def load_calibration(
    yaml_file: str,
) -> Tuple[
    npt.NDArray[np.float64], npt.NDArray[np.float64], Tuple[int, int], Dict[str, Any]
]:
    """Load calibration parameters from YAML file."""
    with open(yaml_file, "r") as f:
        calib: Dict[str, Any] = yaml.safe_load(f)

    # Reconstruct camera matrix
    K = np.array(calib["camera_matrix"]["data"], dtype=np.float64).reshape(3, 3)

    # Get distortion coefficients
    dist = np.array(calib["distortion_coefficients"]["data"], dtype=np.float64)

    # Get original image size
    width: int = calib["image_size"]["width"]
    height: int = calib["image_size"]["height"]

    return K, dist, (width, height), calib


def scale_intrinsics(
    K_original: npt.NDArray[np.float64],
    original_size: Tuple[int, int],
    target_size: Tuple[int, int],
    crop_scale: float,
) -> npt.NDArray[np.float64]:
    """Scale camera intrinsics for a different resolution with cropping.

    Args:
        K_original: Original 3x3 camera matrix
        original_size: (width, height) of original calibration
        target_size: (width, height) of target resolution
        crop_scale: Scale factor due to digital zoom/crop (>1 means more zoomed in)

    Returns:
        K_scaled: Adjusted camera matrix for target resolution

    """
    K_scaled: npt.NDArray[np.float64] = K_original.copy()

    orig_w, orig_h = original_size
    target_w, target_h = target_size

    # Extract original parameters
    fx = K_original[0, 0]
    fy = K_original[1, 1]
    cx = K_original[0, 2]
    cy = K_original[1, 2]

    # Focal length scaling has two components:
    # 1. Resolution scaling: focal length in pixels scales with image dimensions
    # 2. Crop/zoom scaling: cropping increases effective focal length

    resolution_scale_x = target_w / orig_w
    resolution_scale_y = target_h / orig_h

    fx_scaled = fx * resolution_scale_x * crop_scale
    fy_scaled = fy * resolution_scale_y * crop_scale

    # Principal point scales with resolution
    # For centered crop, it stays at the image center after scaling
    cx_scaled = (cx / orig_w) * target_w
    cy_scaled = (cy / orig_h) * target_h

    K_scaled[0, 0] = fx_scaled
    K_scaled[1, 1] = fy_scaled
    K_scaled[0, 2] = cx_scaled
    K_scaled[1, 2] = cy_scaled

    return K_scaled


def generate_scaled_calibrations(
    calibration_file: str, output_dir: str = "."
) -> List[str]:
    """Generate scaled calibration files for all resolution modes."""
    print(f"Loading calibration from {calibration_file}...")
    K_original, dist_original, original_size, original_calib = load_calibration(
        calibration_file
    )

    print("\nOriginal calibration:")
    print(f"  Resolution: {original_size[0]}x{original_size[1]}")
    print(f"  fx={K_original[0, 0]:.2f}, fy={K_original[1, 1]:.2f}")
    print(f"  cx={K_original[0, 2]:.2f}, cy={K_original[1, 2]:.2f}")
    print(f"  RMS error: {original_calib.get('rms_error', 'N/A')}")

    print(f"\n{'=' * 70}")
    print("GENERATING SCALED CALIBRATIONS")
    print(f"{'=' * 70}\n")

    generated_files = []

    for mode_name, mode_info in RESOLUTION_MODES.items():
        target_size = mode_info["resolution"]
        crop_scale = mode_info["crop_scale"]
        description = mode_info["description"]
        assert isinstance(target_size, tuple) and len(target_size) == 2
        assert isinstance(crop_scale, float)
        assert isinstance(description, str)

        print(f"{mode_name}: {description}")
        print(f"  Target resolution: {target_size[0]}x{target_size[1]}")
        print(f"  Crop scale: {crop_scale:.3f}x")

        # Scale intrinsics
        K_scaled = scale_intrinsics(K_original, original_size, target_size, crop_scale)

        print("  Scaled parameters:")
        print(f"    fx={K_scaled[0, 0]:.2f}, fy={K_scaled[1, 1]:.2f}")
        print(f"    cx={K_scaled[0, 2]:.2f}, cy={K_scaled[1, 2]:.2f}")

        # Create calibration dict for this mode
        scaled_calib = {
            "calibration_date": original_calib["calibration_date"],
            "source_calibration": calibration_file,
            "resolution_mode": mode_name,
            "description": description,
            "image_size": {"width": int(target_size[0]), "height": int(target_size[1])},
            "camera_matrix": {
                "rows": 3,
                "cols": 3,
                "data": K_scaled.flatten().tolist(),
            },
            "distortion_coefficients": {
                "rows": 1,
                "cols": len(dist_original),
                "data": dist_original.tolist(),
            },
            "distortion_model": original_calib["distortion_model"],
            "crop_scale": float(crop_scale),
            "fx": float(K_scaled[0, 0]),
            "fy": float(K_scaled[1, 1]),
            "cx": float(K_scaled[0, 2]),
            "cy": float(K_scaled[1, 2]),
        }

        # Save to file
        output_file = f"{output_dir}/calibration_{mode_name}.yaml"
        with open(output_file, "w") as f:
            yaml.dump(scaled_calib, f, default_flow_style=False, sort_keys=False)

        generated_files.append(output_file)
        print(f"  Saved to: {output_file}\n")

    print(f"{'=' * 70}")
    print(f"Generated {len(generated_files)} calibration files")
    print(f"{'=' * 70}")

    return generated_files


def main() -> None:
    """Run the scale calibration tool."""
    parser = argparse.ArgumentParser(
        description="Scale camera calibration for different resolution modes"
    )
    parser.add_argument(
        "calibration_file",
        type=str,
        help="Input calibration YAML file (from calibrate.py)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=".",
        help="Output directory for scaled calibration files (default: current directory)",
    )

    args = parser.parse_args()

    try:
        generated_files = generate_scaled_calibrations(
            args.calibration_file, args.output_dir
        )

        print("\n✓ Success! Generated calibration files:")
        for f in generated_files:
            print(f"  - {f}")

    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
