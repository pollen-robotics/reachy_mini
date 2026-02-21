# Reachy Mini -- Calibration Guide

Make precision feel achievable. Know when calibration is needed and how to fix drift.

---

## When Calibration Is Required

Reachy Mini's Dynamixel motors use absolute encoders, so **you typically do not need to calibrate joint positions.** The motors know where they are after power-on.

However, calibration is relevant in these situations:

| Situation | What to do |
|-----------|-----------|
| **Camera image is dark** | Adjust exposure (see below) |
| **Motor shows wrong position after reassembly** | Check motor orientation marks |
| **Head drifts or feels off-center** | Verify motor ID mapping and physical alignment |
| **Audio direction-of-arrival is inaccurate** | Microphone array calibration (firmware-level) |
| **Antenna appears rotated 90/180 degrees** | Physical repositioning (manufacturing offset) |

---

## Camera Exposure Calibration

The most common "calibration" task. If the camera image appears dark:

### Quick Fix (Any OS)
Enable auto-exposure or increase exposure time using a camera control app:

| OS | Application |
|----|------------|
| **macOS** | [CameraController](https://github.com/itaybre/CameraController) |
| **Linux** | `qv4l2` (install: `sudo apt install qv4l2`) |
| **Windows** | [Webcam Settings](https://www.softpedia.com/get/Internet/WebCam/Webcam-Settings-Tool.shtml) |

### Programmatic Fix (Linux)
```bash
# Install v4l2 utilities
sudo apt install v4l-utils

# List available controls
v4l2-ctl --list-ctrls

# Set auto-exposure priority (fixes darkness)
v4l2-ctl --set-ctrl=auto_exposure_priority=1
```

### From Python (OpenCV)
```python
import cv2

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 3)  # Auto exposure on
# Or manual:
# cap.set(cv2.CAP_PROP_EXPOSURE, 200)
```

---

## Motor Alignment Verification

After assembly or reassembly, verify motors are correctly oriented:

### Step 1: Scan Motors
```bash
# On Lite (from the tools/ directory):
python src/reachy_mini/tools/scan_motors.py

# On Wireless (SSH in first):
ssh pollen@reachy-mini.local  # password: root
source /venvs/mini_daemon/bin/activate
python scan_motors.py
```

Expected output -- all 9 motors on baudrate 1,000,000:
```
Found motors at baudrate 1000000: [10, 11, 12, 13, 14, 15, 16, 17, 18]
```

### Step 2: Check Motor IDs

| Expected ID | Motor |
|-------------|-------|
| 10 | body_rotation |
| 11-16 | stewart_1 through stewart_6 |
| 17 | right_antenna |
| 18 | left_antenna |

If a motor has the wrong ID or baudrate, use the **Reachy Mini Testbench app** or the reflash tool:
```bash
reachy-mini-reflash-motors
```

### Step 3: Verify Physical Alignment
Each motor has an **orientation mark**. During assembly, these marks must be aligned. If marks are misaligned, the motor will report incorrect positions and may trigger "Overload Error."

---

## Antenna Repositioning

If an antenna appears physically rotated (90 or 180 degrees off from where it should be):

1. This is a **manufacturing offset**, not a software issue.
2. Follow the [antenna repositioning guide](https://drive.google.com/file/d/1FsmNpwELuXUbdhGHDMjG_CNpYXOMtR7A/view).
3. It involves loosening the antenna, rotating it to the correct position, and re-tightening.

---

## PID Tuning (Advanced)

If a motor is shaky or jittery (especially motors 10, 17, 18), you can tune the PID values:

**Default PID values:**

| Motor | P | I | D |
|-------|---|---|---|
| body_rotation (10) | 200 | 0 | 0 |
| stewart_1-6 (11-16) | 300 | 0 | 0 |
| antennas (17, 18) | 200 | 0 | 0 |

**To reduce jitter:**
1. Lower P to ~180 on the affected motor.
2. If still jittery, increase D to ~10.

PID values are configured in `src/reachy_mini/assets/config/hardware_config.yaml`.

---

## After Transport

After transporting the robot:

1. **Visual inspection:** Check that no cables are disconnected or pinched.
2. **Power on** and open the dashboard to verify all motors are detected.
3. **Run a basic motion test:**
   ```bash
   python examples/minimal_demo.py
   ```
4. If motors are missing, check the physical connections at the foot PCB and head PCB.

---

## Symptoms of Bad Calibration / Alignment

| Symptom | Likely Cause |
|---------|-------------|
| Motor blinks red on startup | Orientation mark misaligned or motor overloaded |
| Head drifts to one side at rest | One Stewart motor may have incorrect ID or position |
| Antenna faces the wrong way | Manufacturing offset (reposition physically) |
| Camera image consistently dark | Auto-exposure disabled or wrong setting |
| Audio direction-of-arrival always off | Flat flex cable installed backwards |

---

## Getting Help

If basic checks do not resolve the issue:
1. Run the **Reachy Mini Testbench app** from the dashboard to scan and diagnose motors.
2. Check `docs/source/platforms/reachy_mini/motors_diagnosis.md` for the full motor troubleshooting flowchart.
3. Ask on Discord: https://discord.gg/Y7FgMqHsub
