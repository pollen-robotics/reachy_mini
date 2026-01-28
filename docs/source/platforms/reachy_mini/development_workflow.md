# Development Workflow for Wireless Reachy Mini

This guide covers efficient workflows for developing and testing code on the Wireless Reachy Mini.

## Prerequisites

- SSH access to your robot (`ssh pollen@reachy-mini.local`, password: `root`)
- SSHFS installed on your computer (`sudo apt install sshfs` on Ubuntu/Debian)
- Your robot's IP address (find it in the dashboard, router, or run `ifconfig` after SSH)

## Quick Cross-Platform Options

Before diving into the full workflows, here are two simpler cross-platform approaches:

### VS Code Remote SSH

VS Code's [Remote - SSH extension](https://marketplace.visualstudio.com/items?itemName=ms-vscode-remote.remote-ssh) lets you edit files directly on the robot. Connect to `pollen@reachy-mini.local`, then open any folder. Changes are saved directly on the robot. This works on Windows, macOS, and Linux.

### Rsync

Use `rsync` to sync your local source code to the robot's site-packages. This is fast and works (almost) everywhere:

```bash
rsync -avz /path/to/your_app/src/your_app/ \
    pollen@reachy-mini.local:/venvs/apps_venv/lib/python3.12/site-packages/your_app/
```

Run this after each edit to push changes. Add `--delete` to remove files that no longer exist locally.

For both options, see [Step 3 of Approach A](#step-3-install-and-run-your-code-on-the-robot) to run your code on the robot.

---

## Approach A: Clone on Robot, Edit Locally (Recommended)

This is the preferred workflow. Your code lives on the robot, but you edit it from your computer using your favorite IDE or AI coding tools.

### Step 1: Clone your repository on the robot

```bash
ssh pollen@reachy-mini.local
cd /home/pollen
git clone https://github.com/YOUR_USER/YOUR_APP.git
```

### Step 2: Mount the robot's files to your local machine

On your local computer, create a mount point and mount:

```bash
mkdir -p ~/wireless_dev
sshfs pollen@reachy-mini.local:/home/pollen/YOUR_APP ~/wireless_dev \
    -o reconnect,ServerAliveInterval=15,ServerAliveCountMax=3
```

Now you can open `~/wireless_dev` in your IDE and edit as if the files were local.

### Step 3: Install and run your code on the robot

SSH into the robot and install/run your app:

```bash
ssh pollen@reachy-mini.local
cd /home/pollen/YOUR_APP

# Install in editable mode (changes apply immediately):
/venvs/apps_venv/bin/pip install -e .

# Then run your app:
/venvs/apps_venv/bin/python -m YOUR_MODULE.main

# Or run directly without installing:
/venvs/apps_venv/bin/python your_script.py
```

### Step 4: To unmount when done

```bash
fusermount -u ~/wireless_dev
```

## Approach B: Override Installed App Sources

If you already installed an app via the dashboard and want to modify its source code directly by mounting your local files onto the robot.

### Step 1: Locate the installed app on the robot

Apps are installed in:
```
/venvs/apps_venv/lib/python3.12/site-packages/YOUR_APP_NAME/
```

### Step 2: Mount your local source over the installed location

Run this **on the robot** to mount your PC's source code over the installed package:

```bash
ssh pollen@reachy-mini.local

# Mount your local src content onto site-packages
sshfs YOUR_USER@YOUR_PC_IP:/path/to/your_app/src/your_app \
    /venvs/apps_venv/lib/python3.12/site-packages/YOUR_APP_NAME \
    -o reconnect,ServerAliveInterval=15,ServerAliveCountMax=3
```

Important: Mount only the **content of your `src/your_app/` directory**, not the entire repository. The site-packages directory contains the package directly, not the repo structure.

Now edit files on your PC. Changes apply immediately when you restart the app.

## Approach C: Mount Local Source and Run Directly

Similar to Approach B, but without using pip install or the dashboard. You mount your local source onto the robot and run the app directly.

### Step 1: Mount your local source onto the robot

Run this **on the robot**:

```bash
ssh pollen@reachy-mini.local
mkdir -p /home/pollen/my_app_mount

sshfs YOUR_USER@YOUR_PC_IP:/path/to/your_app /home/pollen/my_app_mount \
    -o reconnect,ServerAliveInterval=15,ServerAliveCountMax=3
```

### Step 2: Run your app directly

```bash
cd /home/pollen/my_app_mount
/venvs/apps_venv/bin/python main.py
```

This approach is fast for testing, but you won't have the app registered in the dashboard.

## Installing a Specific Branch or Version

To install a specific version system-wide:

```bash
ssh pollen@reachy-mini.local
/venvs/apps_venv/bin/python -m pip install --force-reinstall \
    "git+https://github.com/pollen-robotics/MY_AWESOME_APP.git@MY_AWESOME_BRANCH"
```

Replace `MY_AWESOME_BRANCH` with a branch name (`develop`), tag, or commit hash.

## Common Pitfalls

### Slow pip install over SSHFS

If you mount files **from your computer to the robot** (inverse of Approach A), pip install operations will be extremely slow because pip reads many small files over the network.

**Solution:** Either:
- Use Approach A (files live on robot, mount to your computer)
- Or skip pip install and run manually with `python -m your_module`

### Wrong mount point for site-packages

A git repository has this structure:
```
your_app/
  src/
    your_app/
      __init__.py
      main.py
```

But site-packages contains:
```
your_app/
  __init__.py
  main.py
```

If you mount the whole repo over site-packages, Python will not find your code. Mount only the inner package directory.


## Quick Reference

| Task | Command |
|------|---------|
| SSH to robot | `ssh pollen@reachy-mini.local` |
| Stop daemon | `sudo systemctl stop reachy-mini-daemon` |
| Start daemon | `sudo systemctl start reachy-mini-daemon` |
| View daemon logs | `journalctl -u reachy-mini-daemon -f` |
| Check robot status | `reachyminios_check` |
| Mount robot files locally | `sshfs pollen@IP:/path ~/local_mount -o reconnect,ServerAliveInterval=15,ServerAliveCountMax=3` |
| Unmount | `fusermount -u ~/local_mount` |

## See Also

- [Install Daemon from Branch](install_daemon_from_branch.md) - For testing unreleased daemon versions
- [Troubleshooting](../../troubleshooting.md) - Common issues and fixes
