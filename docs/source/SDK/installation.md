# 📦 Installation of the Daemon and Python SDK

**Supported OS:** We support and test on **Linux** and **macOS**. It also works on Windows, but it is less tested at the moment. Do not hesitate to open an issue if you encounter any problem.

## 1. Prerequisites

* **Python:** You need Python installed on your computer (versions from **3.10 to 3.13** are supported).
* **Git LFS:** You must have `git-lfs` installed to correctly download model assets.

**Install Git LFS:**
* **Linux:** `sudo apt install git-lfs`
* **macOS:** `brew install git-lfs`
* **Windows:** [Follow the official instructions](https://docs.github.com/en/repositories/working-with-files/managing-large-files/installing-git-large-file-storage?platform=windows)

## 2. Set up a Virtual Environment (Highly Recommended)

We **strongly recommend** using a virtual environment. This isolates the Reachy Mini installation and prevents dependency conflicts with your other Python projects.

**Create and activate the environment:**

* **macOS / Linux:**
    ```bash
    python3 -m venv .venv
    source .venv/bin/activate
    ```

* **Windows:**
    ```powershell
    python -m venv .venv
    .venv\Scripts\activate
    ```

> *Once activated, you should see `(.venv)` appear at the start of your command line.*

## 3. Install the Package

You can install Reachy Mini from PyPI (standard) or from the source code (for development).

**Option A: Install from PyPI (Standard)**
Best for most users who just want to use the robot.

```bash
pip install reachy-mini
```


### Option B: Install from Source (For Developers)

Best if you want to modify the SDK code.

```bash
git clone https://github.com/pollen-robotics/reachy_mini
pip install -e ./reachy_mini
```

> **Note for `uv` users:** You can skip the venv setup and run commands directly with `uv run reachy-mini-daemon`.

### 🐧 Linux Users: USB Permission Setup
If you are on Linux and using the robot via USB, you must set up udev rules:
<details>
<summary>Click to see udev instructions</summary>

```bash
echo 'SUBSYSTEM=="tty", ATTRS{idVendor}=="1a86", ATTRS{idProduct}=="55d3", MODE="0666", GROUP="dialout"
SUBSYSTEM=="tty", ATTRS{idVendor}=="38fb", ATTRS{idProduct}=="1001", MODE="0666", GROUP="dialout"' \
| sudo tee /etc/udev/rules.d/99-reachy-mini.rules

sudo udevadm control --reload-rules && sudo udevadm trigger
sudo usermod -aG dialout $USER
# Log out and log back in!
```
⚠️ Important: You may need to log out and log back in for the group changes to take effect.

</details>

## ❓ Troubleshooting
Encountering an issue? 👉 **[Check the Troubleshooting & FAQ Guide](../troubleshooting.md)**

## Next Steps
* **[Quickstart Guide](quickstart.md)**: Run your first behavior on Reachy Mini
* **[Python SDK](python-sdk.md)**: Learn to move, see, speak, and hear.
* **[AI Integrations](integration.md)**: Connect LLMs, build Apps, and publish to Hugging Face.
* **[Core Concepts](core-concept.md)**: Architecture, coordinate systems, and safety limits.

