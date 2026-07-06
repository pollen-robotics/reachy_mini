"""Detect ArUco markers in still images.

Minimal validation step for a future daemon feature: point a marker cube at
Reachy Mini's camera and trigger predefined behaviors (Wi-Fi setup, emotions,
etc). This script only answers "which markers are visible in these images?".

The marker dictionary is unknown (old printed cube), so by default the script
sweeps every predefined OpenCV dictionary and reports all detections. Once the
dictionary is identified, pass --dict to restrict detection to it.

Usage:
    python detect_aruco.py photo1.png photo2.png
    python detect_aruco.py --dict DICT_4X4_50 photo*.png
    python detect_aruco.py --annotate photo*.png   # saves *_aruco.png copies

Note:
    This example requires the OpenCV optional dependency.
    Install with: pip install reachy_mini[opencv]

"""

import argparse
import sys
from pathlib import Path

try:
    import cv2
except ImportError:
    print("Error: OpenCV is required for this example but not installed.")
    print("Install it with: pip install reachy_mini[opencv]")
    sys.exit(1)


def predefined_dictionaries() -> dict[str, int]:
    """Return all predefined ArUco dictionary names and their enum values."""
    return {
        name: getattr(cv2.aruco, name)
        for name in dir(cv2.aruco)
        if name.startswith("DICT_")
    }


def detect(image, dict_id: int):
    """Detect markers with one dictionary; return (corners, ids)."""
    dictionary = cv2.aruco.getPredefinedDictionary(dict_id)
    detector = cv2.aruco.ArucoDetector(dictionary, cv2.aruco.DetectorParameters())
    corners, ids, _rejected = detector.detectMarkers(image)
    return corners, ids


def main() -> None:
    """Run detection on the given images and print a summary."""
    parser = argparse.ArgumentParser(description="Detect ArUco markers in images.")
    parser.add_argument("images", nargs="+", help="Image files to analyze.")
    parser.add_argument(
        "--dict",
        dest="dict_name",
        help="Restrict to one dictionary (e.g. DICT_4X4_50). Default: try all.",
    )
    parser.add_argument(
        "--annotate",
        action="store_true",
        help="Save a copy of each image with detected markers drawn on it.",
    )
    args = parser.parse_args()

    dictionaries = predefined_dictionaries()
    if args.dict_name is not None:
        if args.dict_name not in dictionaries:
            print(f"Unknown dictionary {args.dict_name}. Choices:")
            print(", ".join(sorted(dictionaries)))
            sys.exit(1)
        dictionaries = {args.dict_name: dictionaries[args.dict_name]}

    for path in args.images:
        image = cv2.imread(path)
        if image is None:
            print(f"{path}: could not read image")
            continue

        hits = []
        for name, dict_id in sorted(dictionaries.items()):
            corners, ids = detect(image, dict_id)
            if ids is not None and len(ids) > 0:
                hits.append((name, corners, ids.flatten().tolist()))

        print(f"{path}:")
        if not hits:
            print("  no markers found")
            continue
        for name, _corners, ids in hits:
            print(f"  {name}: ids {ids}")

        if args.annotate:
            annotated = image.copy()
            for name, corners, ids in hits:
                cv2.aruco.drawDetectedMarkers(annotated, corners)
            out = Path(path).with_stem(Path(path).stem + "_aruco")
            cv2.imwrite(str(out), annotated)
            print(f"  annotated image saved to {out}")


if __name__ == "__main__":
    main()
