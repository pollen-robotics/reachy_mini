# Camera Calibration for Reachy Mini

This directory contains tools for calibrating the Reachy Mini camera using a Charuco board.

## Charuco Board Specifications

- Dictionary: DICT_4X4_1000
- Grid size: 11x8 squares
- Square length: 20.75mm
- Marker length: 15.58mm
- Legacy pattern: Enabled (required for proper detection)

## Calibration Workflow

### 1. Acquire Calibration Images

Use `acquire.py` to capture calibration images at the highest resolution (3840x2592):

```bash
# With display (GUI mode)
python acquire.py

# Without display (headless mode - for Raspberry Pi/wireless version)
python acquire.py --headless
```

**GUI Mode:**
- The camera will automatically use the maximum available resolution
- Move the Charuco board to different positions and angles
- Press **Enter** in the camera window to capture an image (aim for 20-30 images)
- Vary the board position: different distances, angles, and locations in the frame
- Press **'q'** to quit

**Headless Mode (for Raspberry Pi/wireless version):**
- No display required
- Position the board, then press **Enter** in terminal to capture
- Type **'q'** and press Enter to quit
- Images saved with confirmation messages

Images will be saved to `./images/`

### 2. Calibrate the Camera

Run the calibration script to compute camera intrinsics and distortion coefficients:

```bash
python calibrate.py [--images-dir ./images] [--output calibration.yaml] [--visualize]
```

Options:
- `--images-dir`: Directory containing calibration images (default: ./images)
- `--output`: Output calibration file (default: calibration.yaml)
- `--min-markers`: Minimum corners per image (default: 20)
- `--visualize`: Show detected corners during processing

The script will:
- Detect Charuco corners in each image
- Perform camera calibration using rational distortion model
- Save calibration parameters to YAML file
- Display RMS reprojection error (should be < 0.5 pixels for good calibration)

### 3. Scale Calibration for Other Resolutions

The calibration is performed at the highest resolution (3840x2592). Other resolution modes use cropping/digital zoom:

- R3840x2160: ~11% crop (vertical)
- R3264x2448: ~11% crop (both axes)
- R1920x1080: ~11% crop (vertical)

Use the scaling script to generate adjusted calibration files for all resolution modes:

```bash
python scale_calibration.py calibration.yaml [--output-dir .]
```

This will create:
- `calibration_R3840x2592at30fps.yaml` (reference, same as input)
- `calibration_R3840x2160at30fps.yaml` (4K UHD)
- `calibration_R3264x2448at30fps.yaml` (3MP)
- `calibration_R1920x1080at60fps.yaml` (Full HD 60fps)

Each file contains adjusted camera intrinsics (fx, fy, cx, cy) for that resolution mode.
The distortion coefficients remain the same across all modes.

The crop analysis is documented in `images_for_crop_analysis/` and analyzed with `analyze_crop_v3.py`.

### 4. Visualize Undistorted Feed

Test your calibration by viewing the live undistorted camera feed:

```bash
python visualize_undistorted.py [--resolution R1920x1080at60fps]
```

The script will:
- Let you choose a resolution mode (or use --resolution to specify)
- Load the corresponding calibration file
- Show side-by-side view of original and undistorted feed
- Press **'s'** to toggle between split view and full undistorted view
- Press **'q'** to quit

This helps verify that:
- Straight lines appear straight after undistortion
- The calibration quality is good (no excessive warping at edges)
- The correct calibration file is being used for each resolution

Note: The undistorted view automatically crops to remove black borders (alpha=0), showing only the valid pixel region.

## Analysis Tools

- **`acquire_crop.py`**: Capture images at different resolutions for crop analysis
  ```bash
  # GUI mode
  python acquire_crop.py

  # Headless mode (for Raspberry Pi/wireless)
  python acquire_crop.py --headless
  ```
  Automatically cycles through all available resolutions and captures one image per resolution.

- **`analyze_crop_v3.py`**: Analyze how much each resolution mode crops the image

- **`compute_crop.py`**: Simple Charuco detection visualization