# Install the Daemon from a Specific Branch

> **⚠️ For Developers and Testers Only**
>
> This guide explains how to install the Reachy Mini daemon from a specific GitHub branch before it is officially released. Use this for testing new features or bug fixes.
>
> **💡 Tip:** Starting from v1.2.14, you can install a branch directly from **Settings → Reachy Mini Update** in the web dashboard.

## Prerequisites

- SSH access to your Reachy Mini robot (`pollen@reachy-mini.local`, password: `root`)
- The robot must be connected to your Wi-Fi network (or accessible through its hotspot)

## Option A: Local Development Setup

> This option is intended for active development and fast debugging cycles. It allows you to safely test changes without affecting the system-wide installation.
> ⚠️ Avoid installing dashboard apps with this option—any changes made to the local `reachy_mini` version won’t be propagated correctly.

### Steps:

1. **Connect to the robot via SSH:**
   ```bash
   ssh pollen@reachy-mini.local
   # Password: root
   ```

2. **Clone the Reachy Mini repository with the specific branch:**
   ```bash
   git clone -b <branch-name> https://github.com/pollen-robotics/reachy_mini.git
   cd reachy_mini
   ```

3. **Set up the virtual environment:**
   ```bash
   uv venv --python /venvs/mini_daemon/bin/python .venv
   source .venv/bin/activate
   uv sync --extra gstreamer --extra wireless-version
   ```

4. **Stop the system daemon service:**
   ```bash
   sudo systemctl stop reachy-mini-daemon
   ```
   **Note:** This step must be repeated after each reboot since the system service restarts automatically.

5. **Start the local daemon for testing:**
   ```bash
   reachy-mini-daemon --wireless-version
   ```

Now you can modify the code in `~/reachy_mini` and test your changes without affecting the system installation.

## Option B: System-Wide Custom Installation

> This option installs a branch build of reachy-mini as the system-wide daemon. It’s better suited for thorough, end-to-end testing and supports seamless app installation from the dashboard.

### Steps:

1. **Connect to the robot via SSH:**
   ```bash
   ssh pollen@reachy-mini.local
   # Password: root
   ```

2. **Activate the daemon's virtual environment:**
   ```bash
   source /venvs/mini_daemon/bin/activate
   ```

3. **Install the specific branch:**
   ```bash
   pip install --no-cache-dir --force-reinstall \
     "reachy_mini[gstreamer,wireless-version] @ git+https://github.com/pollen-robotics/reachy_mini.git@<branch-name>"
   ```
   Replace `<branch-name>` with the branch you want to test (e.g., `develop`, `feature/my-feature`, `bugfix/issue-123`).

   > _Remark_ : we have to use `pip` here and not `uv` because `uv pip install` [does not work correctly with `git lfs`](https://github.com/astral-sh/uv/issues/3312)

4. **(Only for versions ≤ 1.2.13)** Repeat steps 2 and 3 using `/venvs/apps_venv`.

5. **Restart the daemon service:**
   ```bash
   sudo systemctl restart reachy-mini-daemon
   ```

6. **Verify the installation was successful:**
   ```bash
   pip show reachy-mini | grep Version
   ```
   This should display the version corresponding to your installed branch.

## Rolling Back to Factory Version

If you encounter issues with the branch installation, you can restore the factory daemon:

1. **Trigger the SOFTWARE_RESET command** via Bluetooth to reinstall the original factory daemon
2. **Refer to the [Reset Guide](reset.md)** for detailed step-by-step instructions

## Important Notes

- **Backup your work** before switching between different branch installations
- **Test thoroughly** in local development mode before doing system-wide installations
- **Monitor system logs** after installation: `journalctl -u reachy-mini-daemon -f`
- **Performance impact:** Some development branches may have reduced performance or stability
