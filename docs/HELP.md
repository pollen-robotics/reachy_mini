# Reachy Mini -- First-Run Survival Guide

Get from unboxing to your first successful interaction in under 15 minutes.

---

## What Reachy Mini Is (and Is Not)

Reachy Mini is a small expressive robot with:
- A **6-DOF head** (pitch, roll, yaw + x/y/z translation via a Stewart platform)
- A **body** that rotates around its vertical axis
- **Two antennas** that double as physical buttons

It is **not** a humanoid arm robot. It does not pick things up. It expresses, reacts, looks, listens, and speaks.

---

## Minimum Requirements

| Requirement | Details |
|-------------|---------|
| **Python** | 3.10 -- 3.12 |
| **OS** | Linux, macOS, or Windows |
| **Power** | 7V / 5A power supply (included). USB alone will NOT power the motors. |
| **Connection** | Lite: USB-C cable. Wireless: WiFi on the same network. |
| **Git + Git LFS** | Required to clone repo and download model assets |

---

## Golden Path: Box to First Interaction

### 1. Assemble the robot
Follow the printed booklet or the [interactive digital guide](https://huggingface.co/spaces/pollen-robotics/Reachy_Mini_LITE_Assembly_Guide). Budget 2--3 hours.

### 2. Power up and connect
- Plug in the **power supply** (wall outlet, not USB).
- **Lite:** Connect the USB-C cable to your computer.
- **Wireless:** Power on, connect to `reachy-mini-ap` WiFi (password: `reachy-mini`), then configure your home WiFi at `http://reachy-mini.local:8000/settings`.

### 3. Install the SDK
```bash
# Install uv (fast Python package manager)
curl -LsSf https://astral.sh/uv/install.sh | sh   # Linux/macOS
# Or: powershell -c "irm https://astral.sh/uv/install.ps1 | iex"  # Windows

# Create a virtual environment and install
uv venv reachy_mini_env --python 3.12
source reachy_mini_env/bin/activate   # Linux/macOS
# Or: reachy_mini_env\Scripts\activate  # Windows
uv pip install reachy-mini
```

### 4. Start the daemon
```bash
# Lite (USB):
uv run reachy-mini-daemon

# Simulation (no robot):
uv run reachy-mini-daemon --sim

# Wireless: daemon is already running on the robot.
```

### 5. Verify the dashboard
Open http://localhost:8000 in your browser. If you see the Reachy Dashboard, you are ready.

### 6. Run your first script
Save this as `hello.py` and run `python hello.py`:

```python
from reachy_mini import ReachyMini

with ReachyMini() as mini:
    print("Connected!")
    mini.goto_target(antennas=[0.5, -0.5], duration=0.5)
    mini.goto_target(antennas=[-0.5, 0.5], duration=0.5)
    mini.goto_target(antennas=[0, 0], duration=0.5)
    print("Done!")
```

If the antennas wiggle, everything works.

---

## First-Hour Failures and Fixes

| Symptom | Cause | Fix |
|---------|-------|-----|
| Motors don't move | Power supply not connected | Plug in the 7V wall adapter. USB alone is not enough. |
| `Connection refused` | Daemon not running | Open a terminal and run `reachy-mini-daemon` |
| `PortAudio library not found` | Missing system dependency (Linux) | Run `sudo apt-get install libportaudio2` |
| `Permission denied` on serial (Linux) | USB permissions not set | See HELP_SOFTWARE_SETUP.md, "Linux USB Permissions" section |
| Dashboard not loading | Browser blocking localhost | Check browser privacy settings for local network access |
| Wireless robot not found | Not on the same WiFi network | Connect your computer to the same network as the robot |

---

## What to Read Next

| Goal | File |
|------|------|
| Understand the hardware | [HELP_HARDWARE.md](HELP_HARDWARE.md) |
| Detailed environment setup | [HELP_SOFTWARE_SETUP.md](HELP_SOFTWARE_SETUP.md) |
| Move the robot safely | [HELP_FIRST_MOTION.md](HELP_FIRST_MOTION.md) |
| Understand control modes | [HELP_CONTROL_MODES.md](HELP_CONTROL_MODES.md) |
| Something went wrong | [HELP_WHEN_THINGS_GO_WRONG.md](HELP_WHEN_THINGS_GO_WRONG.md) |

---

## Community and Support

- **Discord:** https://discord.gg/Y7FgMqHsub
- **GitHub Issues:** https://github.com/pollen-robotics/reachy_mini/issues
- **Community Apps:** https://huggingface.co/spaces?q=reachy_mini
