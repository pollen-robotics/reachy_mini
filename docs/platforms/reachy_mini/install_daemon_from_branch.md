# Install the Daemon from a Specific Branch

> **⚠️ Developers/Testers Only**
>
> Use this guide to install the Reachy Mini daemon from a GitHub branch before it is officially released.

## Prerequisites

- SSH access to your Reachy Mini (`pollen@reachy-mini.local`, password `root`).
- The robot connected to your Wi-Fi (or reachable through the hotspot).

## Procedure

1. **SSH into the robot:**
   ```bash
   ssh pollen@reachy-mini.local
   # password: root
   ```

2. **Activate the daemon virtual environment:**
   ```bash
   source /venvs/mini_daemon/bin/activate
   ```

3. **Install the branch:**
   ```bash
   pip install --no-cache-dir --force-reinstall \
     git+https://github.com/pollen-robotics/reachy_mini.git@<branch-name>
   ```
   Replace `<branch-name>` with the branch you want to test (e.g., `develop`, `feature/my-feature`).

4. **Restart the daemon:**
   ```bash
   sudo systemctl restart reachy-mini-daemon
   ```

5. **Verify the installation:**
   ```bash
   pip show reachy-mini | grep Version
   ```

## Rollback to Factory Version

If the branch causes issues, trigger the **SOFTWARE_RESET** command (via Bluetooth) to reinstall the factory daemon. See the [Reset Guide](reset.md) for detailed steps.
