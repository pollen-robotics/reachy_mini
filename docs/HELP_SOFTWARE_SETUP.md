# Reachy Mini -- Software Setup Guide

Eliminate environment hell. Get a clean, working installation on any OS.

---

## Prerequisites

| Tool | Version | Check | Purpose |
|------|---------|-------|---------|
| **Python** | 3.10 -- 3.12 | `python --version` | Run SDK and apps |
| **Git** | Latest | `git --version` | Clone repositories |
| **Git LFS** | Latest | `git lfs version` | Download model assets |
| **uv** (recommended) | Latest | `uv --version` | Fast package management |

---

## Step 1: Install uv (Recommended Package Manager)

**Linux / macOS:**
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

**Windows (PowerShell):**
```powershell
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

Close and reopen your terminal, then verify:
```bash
uv --version
```

---

## Step 2: Install Python

```bash
uv python install 3.12 --default
```

Verify:
```bash
python --version
# Should show Python 3.12.x
```

---

## Step 3: Install Git and Git LFS

**Linux:**
```bash
sudo apt install git git-lfs
```

**macOS:**
```bash
brew install git git-lfs
```

**Windows:** Download from https://git-scm.com/install/windows

Then initialize Git LFS:
```bash
git lfs install
```

---

## Step 4: Create a Virtual Environment

```bash
uv venv reachy_mini_env --python 3.12
```

Activate it:
```bash
# Linux / macOS:
source reachy_mini_env/bin/activate

# Windows:
reachy_mini_env\Scripts\activate
```

You should see `(reachy_mini_env)` at the start of your prompt.

**Important:** You must activate this environment every time you open a new terminal.

---

## Step 5: Install Reachy Mini

**Standard install (most users):**
```bash
uv pip install reachy-mini
```

**With simulation support:**
```bash
uv pip install "reachy-mini[mujoco]"
```

**From source (developers):**
```bash
git clone https://github.com/pollen-robotics/reachy_mini && cd reachy_mini
uv sync
# Or with simulation:
uv sync --extra mujoco
```

---

## Step 6: Linux USB Permissions

If you are using a Lite (USB-connected) robot on Linux, you need to grant serial port access:

```bash
echo 'SUBSYSTEM=="usb", ATTRS{idVendor}=="1a86", ATTRS{idProduct}=="55d3", MODE="0666", GROUP="dialout"
SUBSYSTEM=="usb", ATTRS{idVendor}=="38fb", ATTRS{idProduct}=="1001", MODE="0666", GROUP="dialout"' \
| sudo tee /etc/udev/rules.d/99-reachy-mini.rules

sudo udevadm control --reload-rules && sudo udevadm trigger
sudo usermod -aG dialout $USER
```

**Log out and log back in** for group changes to take effect.

---

## Step 7: Linux Audio Dependency

```bash
sudo apt-get install libportaudio2
```

Without this, you will get `OSError: PortAudio library not found` when the SDK tries to initialize audio.

---

## OS-Specific Notes

### macOS
- MuJoCo simulation requires `mjpython` launcher:
  ```bash
  mjpython -m reachy_mini.daemon.app.main --sim
  ```
- `uv` may have compatibility issues with MuJoCo on macOS. Use `pip` as a fallback.

### Windows
- Before activating a virtual environment for the first time, enable script execution:
  ```powershell
  Set-ExecutionPolicy RemoteSigned
  ```
  (Run this once in an Administrator PowerShell.)
- App installations from Hugging Face may fail due to symlink permissions. Set:
  ```powershell
  set HF_HUB_DISABLE_SYMLINKS_WARNING=1
  ```

---

## Health Check Script

After installation, run this to verify everything works:

```bash
python -c "
from reachy_mini import ReachyMini
print('SDK import: OK')
from reachy_mini.utils import create_head_pose
print('Utilities import: OK')
import numpy as np
pose = create_head_pose(yaw=10, pitch=5, degrees=True)
print('Head pose creation: OK')
print('All checks passed.')
"
```

If any import fails, your virtual environment is not correctly set up or the package is not installed.

---

## Connecting to the Robot

| Setup | Daemon Command | Dashboard URL |
|-------|----------------|---------------|
| **Lite (USB)** | `reachy-mini-daemon` | http://localhost:8000 |
| **Wireless** | Already running on robot | http://reachy-mini.local:8000 |
| **Simulation** | `reachy-mini-daemon --sim` | http://localhost:8000 |

The daemon must be running in a terminal before any Python scripts will work. Keep that terminal open.

---

## Upgrading

```bash
# Via uv:
uv pip install -U reachy-mini

# Via pip:
pip install -U reachy-mini
```

For Wireless robots, use the dashboard Settings page to check for system updates.
