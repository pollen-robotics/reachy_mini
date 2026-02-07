# Reachy Mini -- When Things Go Wrong

Something is not working. That is okay. This guide will help you figure out what happened and get back on track.

---

## "The robot doesn't move"

This is the most common first-time issue. Work through these checks in order:

### Check 1: Is the power supply plugged in?
The USB-C cable provides data only. Motors require the 7V wall adapter.

**Test:** Is the power LED on? If no, plug in the power supply.

### Check 2: Is the daemon running?
- **Lite:** Open a terminal and run `reachy-mini-daemon`. Keep it running.
- **Wireless:** The daemon starts automatically when powered on.
- **Test:** Open http://localhost:8000 (Lite) or http://reachy-mini.local:8000 (Wireless). If the dashboard loads, the daemon is running.

### Check 3: Is another app already controlling the robot?
Only one client can control the robot at a time. Stop any running apps from the dashboard before running your script.

### Check 4: Are motors enabled?
```python
mini.enable_motors()
```
Motors start disabled on some initialization paths. Explicitly enable them.

### Check 5: Is your virtual environment activated?
Look for `(reachy_mini_env)` at the start of your terminal prompt. If missing:
```bash
source reachy_mini_env/bin/activate
```

---

## "The robot moves weirdly"

### Jerky / stuttering motion
- Are you calling `set_target()` from multiple threads? Use a single control loop.
- Is your control loop running below 30Hz? Speed it up to 50--100Hz.
- Are you mixing `goto_target()` and `set_target()`? Pick one per behavior phase.
- Check the motor control loop frequency:
  ```python
  print(mini.client.get_status())
  ```
  It should show ~50Hz (~20ms period). Much higher means CPU or USB latency issues.

### Head jumps when motors are enabled
You enabled motors without first setting the goal to the current position. Use the safe enable pattern:
```python
head_pose = mini.get_current_head_pose()
_, antennas = mini.get_current_joint_positions()
mini.goto_target(head=head_pose, antennas=list(antennas) if antennas else None, duration=0.05)
mini.disable_motors()
mini.enable_motors()
```

### Motor is shaky when holding position
PID values may need tuning. In `src/reachy_mini/assets/config/hardware_config.yaml`, try lowering P to ~180 on the affected motor (commonly 10, 17, or 18).

### Head drifts to one side
Check that all Stewart motors (11--16) are detected and have correct IDs. Run the motor scan script.

---

## "The robot worked yesterday"

### Step 1: Update and restart
The single most effective fix:
- **Wireless:** Press OFF, wait 5 seconds, press ON. Then check for updates in dashboard Settings.
- **Lite:** Update the SDK: `uv pip install -U reachy-mini`. Restart the daemon.

### Step 2: Check what changed
```bash
# If your code is in Git:
git log --oneline -5
git diff HEAD~1
```

Did you change your code? Did you update a dependency? Did you change your Python environment?

### Step 3: Run the minimal demo
```bash
python examples/minimal_demo.py
```
If this works, the problem is in your app code, not the robot or SDK.

### Step 4: Check hardware
- Are all cables still connected? Cables can work loose, especially in the head.
- Is the power supply providing consistent power? Try a different outlet.

---

## "I think I broke it"

Take a breath. Reachy Mini is more resilient than it looks.

### Motor blinking red
This usually means the motor detected an overload, not that it is broken. Try:
1. Power off completely.
2. Wait 30 seconds.
3. Power on.
4. Run the Testbench app from the dashboard to check motor status.

### A motor won't respond
1. Check the cable to that motor.
2. Run `reachy-mini-reflash-motors` to reset the motor firmware.
3. If the motor feels physically stuck and hard to turn by hand (when powered off), it may be a hardware defect. Contact support.

### An antenna is pointing the wrong way
This is a known manufacturing variation. It is not broken -- the antenna was mounted at a different offset. Follow the [antenna repositioning guide](https://drive.google.com/file/d/1FsmNpwELuXUbdhGHDMjG_CNpYXOMtR7A/view).

### Head dropped suddenly
If you called `disable_motors()` or the power was interrupted, the head will drop under gravity. This is normal. It does not damage the robot.

### Something smells like it's burning
**Power off immediately and unplug.** This is not normal. Do not power on again until you have inspected all cables. Contact support if you find damage.

---

## "My code keeps crashing"

### Debug checklist
- [ ] Daemon is running
- [ ] No other apps are connected
- [ ] Virtual environment is activated
- [ ] `reachy-mini` is installed in the current environment
- [ ] Basic connection test passes: `python -c "from reachy_mini import ReachyMini; print('OK')"`
- [ ] Minimal demo runs: `python examples/minimal_demo.py`
- [ ] Motors are enabled

### Common code issues

| Error | Cause | Fix |
|-------|-------|-----|
| `ModuleNotFoundError` | Wrong venv or not installed | Activate venv, `uv pip install reachy-mini` |
| `ConnectionError` | Daemon not running | Start daemon |
| `ValueError` on motion | Wrong input format | Use `create_head_pose()`, pass radians for antennas |
| `TypeError` | Wrong argument types | Check function signatures in SDK docs |

### Read the traceback
Python tracebacks read bottom-to-top. The last line is the actual error. The lines above show where in your code it happened. Start from the bottom.

---

## Recovery Procedures

### Full reset (Wireless)
If nothing else works:
1. SSH in: `ssh pollen@reachy-mini.local` (password: `root`)
2. Restart the daemon: `systemctl restart reachy-mini-daemon.service`
3. If that doesn't help, reboot: `sudo reboot`
4. Last resort: [Reflash the Raspberry Pi ISO](../source/platforms/reachy_mini/reflash_the_rpi_ISO.md)

### Full reset (Lite)
1. Close the daemon.
2. Unplug USB and power.
3. Wait 10 seconds.
4. Reconnect power, then USB.
5. Start the daemon.
6. Test with `python examples/minimal_demo.py`.

### Reinstall the SDK
```bash
uv pip uninstall reachy-mini
uv pip install reachy-mini
```

---

## Getting Human Help

If you have worked through this guide and are still stuck:

1. **Discord** (fastest): https://discord.gg/Y7FgMqHsub
2. **GitHub Issues:** https://github.com/pollen-robotics/reachy_mini/issues

When asking for help, include:
- What you expected to happen
- What actually happened
- The full error message / traceback
- Your OS, Python version, and SDK version (`pip show reachy-mini`)
- Whether the minimal demo (`examples/minimal_demo.py`) works
- Lite or Wireless version
