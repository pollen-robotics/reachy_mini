# Reachy Mini -- Hardware Guide

Prevent physical damage. Understand what you are working with.

---

## Specifications at a Glance

| Spec | Value |
|------|-------|
| **Dimensions** | 30 x 20 x 15.5 cm (extended) |
| **Mass** | 1.35 kg (Lite) / 1.475 kg (Wireless) |
| **Materials** | ABS, PC, Aluminium, Steel |
| **Power input** | 6.8 -- 7.6V DC |
| **Camera** | Sony IMX708, 12MP, 120 degree wide angle, autofocus |
| **Microphones** | 4-mic MEMS array (Seeed reSpeaker XMOS XVF3800), 16 kHz |
| **Speaker** | 5W @ 4 Ohms |
| **Compute (Wireless)** | Raspberry Pi CM4 (4GB RAM, 16GB flash, WiFi) |
| **Battery (Wireless)** | LiFePO4, 2000mAh, 6.4V, 12.8Wh |

---

## Degrees of Freedom

| Component | DOF | Details |
|-----------|-----|---------|
| **Head** | 6 | 3 rotations (pitch, roll, yaw) + 3 translations (x, y, z) via Stewart platform |
| **Body** | 1 | Rotation around vertical axis |
| **Antennas** | 2 | 1 rotation each (right and left) |
| **Total** | 9 | |

---

## Motor Details

| Motor | ID | Type | Purpose |
|-------|----|------|---------|
| `body_rotation` | 10 | XC330-M288-PG | Base rotation |
| `stewart_1` | 11 | XL330-M288-T | Head platform |
| `stewart_2` | 12 | XL330-M288-T | Head platform |
| `stewart_3` | 13 | XL330-M288-T | Head platform |
| `stewart_4` | 14 | XL330-M288-T | Head platform |
| `stewart_5` | 15 | XL330-M288-T | Head platform |
| `stewart_6` | 16 | XL330-M288-T | Head platform |
| `right_antenna` | 17 | XL330-M077-T | Right antenna |
| `left_antenna` | 18 | XL330-M077-T | Left antenna |

Serial baudrate: 1,000,000 bps.

---

## Power

- **Input voltage:** 6.8 -- 7.6V via the included power adapter.
- **USB-C does NOT power the motors.** You must use the wall adapter for any motion.
- **Wireless battery:** LiFePO4 with overcharge, overdischarge, overcurrent, and short-circuit protection. Built-in temperature sensor.
- **Battery indicator:** LED color (green -> orange -> red). There is no precise battery percentage readout.

---

## Safe Boot and Shutdown

### Power On
1. Plug in the power adapter (Lite) or press the ON button (Wireless).
2. Wait for the daemon to start (Wireless: ~30 seconds).
3. Verify via the dashboard at `http://localhost:8000` (Lite) or `http://reachy-mini.local:8000` (Wireless).

### Power Off
1. **Wireless:** Press the OFF button. Wait 5 seconds before unplugging.
2. **Lite:** Close the daemon process, then unplug USB and power.

### Restart (fixes many issues)
Press OFF, wait 5 seconds, press ON.

---

## Cable and Assembly Notes

- **USB cable in the head:** Leave enough slack for full head rotation. A tight cable will restrict motion and eventually fatigue the connector.
- **Motor orientation marks:** Each motor has an alignment mark that must match during assembly. Misalignment causes "Overload Error" on startup.
- **Flat flex cable (mic array):** Must be installed the correct way (see assembly guide slides 45--47). Reversed connection = no microphone input.
- **Antenna positioning:** If an antenna appears rotated 90 or 180 degrees, it is a manufacturing offset. Follow the [antenna repositioning guide](https://drive.google.com/file/d/1FsmNpwELuXUbdhGHDMjG_CNpYXOMtR7A/view) to correct it.

---

## DO NOT Do This

- **DO NOT** power motors from USB alone. They will not respond, and you may think something is broken.
- **DO NOT** force the head past its physical stops. The Stewart platform has mechanical limits.
- **DO NOT** leave the robot powered with motors enabled and unattended for extended periods. Motors may overheat.
- **DO NOT** disconnect cables while the robot is powered on.
- **DO NOT** expose to liquids. No ingress protection rating.
- **DO NOT** use a power supply outside the 6.8 -- 7.6V range. Higher voltage will trigger "Input Voltage Error" on motors (which is intentionally suppressed in the firmware since the robot operates near the upper limit).

---

## Storage and Transport

- Power off and unplug before moving.
- Store in a dry environment at room temperature.
- Use the original packaging for transport when possible.
- The head can rest in its natural sleep position during storage.

---

## Wireless-Specific Notes

- **Switch position on the head board:** Must be in "debug" position, not "download". If the WiFi access point does not appear, check this switch.
- **SSH access:** `ssh pollen@reachy-mini.local` (password: `root`).
- **System health check:** Run `reachyminios_check` after SSH login.
- **USB-C on Wireless:** This is an output port (for USB devices like a flash drive). It does NOT provide a tethered connection like the Lite version.
