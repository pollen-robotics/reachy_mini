# Reachy Mini -- Safety Guide

Reduce risk of injury and equipment damage. Build trust through transparency.

---

## Physical Safety

### Pinch Points

The Stewart platform mechanism (6 actuators connecting body to head) creates pinch points during motion. Keep fingers away from the gap between head and body while the robot is active.

**Specific locations:**
- Between the 6 Stewart platform rods and the head/body shells
- Between the body shell and the base during body rotation
- Near antenna joints when antennas are moving

### For Children and Pets

- **Supervise** children under 12 when the robot is powered on.
- Antennas are lightweight and cannot cause injury, but small children may pull on them.
- The robot is not waterproof or food-safe. Keep away from liquids.
- The power adapter cable is a trip hazard. Route it safely.

---

## Emergency Stop

### From software:
```python
mini.disable_motors()
```
This immediately removes torque from all motors. The head will drop under gravity to its rest position.

### From the dashboard:
Navigate to http://localhost:8000 and disable motors through the interface.

### Physical:
Unplug the power adapter. All motors instantly lose torque.

**Note:** Disabling motors is always safe. The head dropping to rest position is normal and will not damage the robot. Gentle collisions between head and body are expected and harmless.

---

## Operating Limits

### Joint Limits (enforced by software)

| Joint | Range |
|-------|-------|
| Head pitch | -40 to +40 degrees |
| Head roll | -40 to +40 degrees |
| Head yaw | -180 to +180 degrees |
| Body yaw | -160 to +160 degrees |
| Head-body yaw difference | Max 65 degrees |

The SDK automatically clamps values to these ranges. You cannot command the robot past these limits through the API.

### Electrical Limits

| Parameter | Value |
|-----------|-------|
| Input voltage | 6.8 -- 7.6V DC |
| Power supply | Use ONLY the included adapter |
| USB-C | Data only (Lite) or output only (Wireless). Does NOT supply motor power. |

### Thermal Limits

Motors have built-in thermal protection. If a motor overheats:
1. It will stop responding (thermal shutdown).
2. Power off the robot.
3. Wait 5+ minutes for cooling.
4. Power back on.

**Prevention:** Avoid running motors at high torque for extended periods without breaks.

---

## Recommended Operating Time

- **Continuous use (active motion):** Up to 2 hours before checking motor temperatures.
- **Idle with motors enabled:** Motors draw current even when holding position. Disable motors during long idle periods.
- **Wireless battery life:** Approximately 1--2 hours depending on activity level. Charge when the LED turns red.

---

## What the Robot Can Safely Do

- Head can touch the body shell during some motions. This is expected.
- Antennas can be pushed by hand (they are semi-compliant). This is by design -- they are used as physical buttons.
- The robot can be moved by hand when motors are disabled or in gravity compensation mode.

---

## What You Should NOT Do

| Action | Risk |
|--------|------|
| Use a power supply above 7.6V | Motor damage, electrical shock errors |
| Run motors continuously for hours at high torque | Overheating, thermal shutdown |
| Force the head past its physical stops | Mechanical damage to Stewart platform |
| Disconnect cables while powered on | Motor errors, potential damage |
| Operate in wet or humid conditions | No ingress protection |
| Leave unattended with motors enabled and active motion | Overheating risk |
| Place heavy objects on the head | Stewart platform not designed for external loads |

---

## Power Supply Safety

- Use **only** the included 7V/5A power adapter.
- Ensure the power cable is not pinched or bent sharply.
- Unplug when not in use.
- The Wireless battery has built-in protections (overcharge, overdischarge, overcurrent, short circuit, temperature sensor).
- There is no way to check exact battery percentage. Rely on the LED indicator (green > orange > red).

---

## Handling and Transport

- Power off completely before moving the robot.
- Support the head when moving -- it is the heaviest part relative to its mount.
- Use the original packaging for shipping.
- After transport, run a basic motion test before complex operations.

---

## If Something Seems Wrong

1. **Motors making unusual sounds:** Power off immediately. Check for cables caught in mechanisms.
2. **Burning smell:** Power off immediately. Unplug. Do not power on until inspected.
3. **Motor locked and blinking red:** May be overloaded or defective. See HELP_COMMON_ERRORS.md.
4. **Robot falls over:** The base is designed to be stable, but ensure it is on a flat surface. The robot is not designed for inclined surfaces.

For any safety concern, power off first and assess second.

---

## Contact for Safety Issues

If you believe there is a safety defect:
- Email: sales@pollen-robotics.com
- Include photos and description of the issue
- Include your order/invoice number
