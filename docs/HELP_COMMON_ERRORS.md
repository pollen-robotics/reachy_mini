# Reachy Mini -- Common Errors Guide

Searchable, blunt, fast. Find your error, get the fix, move on.

---

## Connection Errors

### `ConnectionError` / `Connection refused` / Timeout on connect

**What it means:** The SDK cannot reach the daemon.

**Fix:**
1. Is the daemon running? Start it: `reachy-mini-daemon` (Lite) or verify robot is powered on (Wireless).
2. Is another app already connected and controlling the robot? Stop it first.
3. For Wireless: Are your computer and robot on the same WiFi network?
4. Try opening http://localhost:8000 (Lite) or http://reachy-mini.local:8000 (Wireless). If the dashboard loads, the daemon is running.

---

### `OSError: PortAudio library not found`

**What it means:** Missing system audio library on Linux.

**Fix:**
```bash
sudo apt-get install libportaudio2
```
Then restart the daemon.

---

### Dashboard at `http://localhost:8000` doesn't load

**What it means:** Daemon is not running or browser is blocking local access.

**Fix:**
1. Check the daemon terminal for errors.
2. On macOS: Check System Settings > Privacy & Security > Local Network for browser permissions.
3. Make sure you are inside your virtual environment.
4. Update the SDK: `uv pip install -U reachy-mini`

---

## Motor Errors

### Motors don't move / No response

**What it usually means:** Power supply is not connected.

**Fix:** Plug in the 7V wall adapter. USB-C provides data only, not motor power.

---

### Motor blinking red / `Overload Error`

**What it means:** Motor is physically stuck or orientation marks are misaligned.

**Fix:**
1. Check that motor orientation marks are aligned (assembly guide).
2. Use the **Reachy Mini Testbench app** to diagnose.
3. If the motor feels hard to turn even when powered off AND blinks red, it may be a hardware defect. Contact support.

---

### `Motor '<name>' hardware errors: ['Input Voltage Error']`

**What it means:** Nothing wrong. Reachy Mini intentionally operates at the upper voltage range of the Dynamixel motors. This warning is expected and suppressed in normal operation.

**Fix:** No action needed.

---

### `Electrical Shock Error`

**What it means:** Power supply issue or short circuit.

**Fix:**
1. Check all cables from foot PCB to head for damage.
2. Inspect the power cable (black & red).
3. Inspect the 3-wire motor cables (300mm, 200mm, 100mm, 40mm).

---

### Motors stop responding after a while

**What it means:** Thermal protection (overheating) or power issue.

**Fix:**
1. Power off, wait 30 seconds, power on.
2. Check power supply connection.
3. Update the SDK: `pip install -U reachy-mini`
4. If motor LED blinks red, see "Overload Error" above.

---

### Motor is shaky / jittery

**What it means:** PID values are causing overcorrection. Common on motors 10 (body), 17 and 18 (antennas).

**Fix:** Tune PID values in `src/reachy_mini/assets/config/hardware_config.yaml`:
- Reduce P to ~180 on the affected motor.
- If still shaky, increase D to ~10.

---

### `No motor found on port` / Missing motors

**What it means:** Motor is not detected on the serial bus.

**Fix:**
1. Check physical cable connections (especially at foot PCB and head PCB).
2. Run the motor scan script to identify which motors are found.
3. If consecutive motors are missing (e.g., 11-12-13 or 17-18), the issue is likely a disconnected cable to that chain.
4. Use `reachy-mini-reflash-motors` if a motor has the wrong baudrate or ID.

---

## SDK / Python Errors

### `ModuleNotFoundError: No module named 'reachy_mini'`

**What it means:** SDK not installed or wrong virtual environment.

**Fix:**
1. Activate your virtual environment: `source reachy_mini_env/bin/activate`
2. Install: `uv pip install reachy-mini`

---

### `ValueError` on `goto_target` or `set_target`

**What it means:** Invalid input shape or type.

**Fix:**
- `head` must be a 4x4 numpy array (use `create_head_pose()` to build it).
- `antennas` must be a list/tuple of 2 floats (in radians).
- `body_yaw` must be a single float (in radians).
- `duration` must be positive.

---

### `Warning: Circular buffer overrun` (Simulation)

**What it means:** Video frames are being produced but not consumed, filling the buffer.

**Fix:** If you don't need video, initialize with:
```python
with ReachyMini(media_backend="no_media") as mini:
    ...
```

---

## Audio / Video Errors

### Camera image is dark (Lite)

**Fix:** Adjust exposure. See HELP_CALIBRATION.md for detailed instructions per OS.

Quick fix: set `auto-exposure-priority=1` using your OS camera controls.

---

### No microphone input (Wireless)

**What it means:** Firmware too old or flat flex cable installed backwards.

**Fix:**
1. Update to firmware 2.1.3+: run the update script at `src/reachy_mini/assets/firmware/update.sh`.
2. Check flat flex cable orientation (assembly guide slides 45--47).

---

### Audio volume too low

**Fix:** Update to SDK version 1.2.3 or later.

On Linux, also check `alsamixer`:
1. Run `alsamixer`
2. Set PCM1 to 100%
3. Adjust global volume with PCM,0

---

### Face tracking feels slow

**Fix:**
1. Ensure the face is well-lit.
2. The GStreamer backend may have lower latency than the default OpenCV backend.
3. On Wireless, remote (WebRTC) introduces additional latency compared to local execution.

---

## Wireless-Specific Errors

### WiFi access point (`reachy-mini-ap`) doesn't appear

**Fix:** Check the switch on the head board. It must be in "debug" position, not "download."

---

### Can't connect via USB-C cable (Wireless)

**What it means:** Wireless units do NOT expose the robot over USB like the Lite version.

**Fix:** Use WiFi. For a wired connection, use a USB-C-to-Ethernet adapter plus Ethernet cable.

---

### App installations fail on Windows

**Fix:**
```powershell
set HF_HUB_DISABLE_SYMLINKS_WARNING=1
```

---

## When to Escalate

Contact support or ask on [Discord](https://discord.gg/Y7FgMqHsub) when:
- A motor feels physically hard to turn when powered off AND blinks red (hardware defect).
- Cables appear physically damaged.
- The motor scan shows motors at unexpected baudrates or IDs after a fresh assembly.
- The problem persists after updating firmware and SDK to latest versions.
