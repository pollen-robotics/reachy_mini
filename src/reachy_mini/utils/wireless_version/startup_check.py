"""Check and fix ownership of files under /venvs directory.

This module ensures that all files under /venvs are owned by the pollen user.
If any files are not owned by pollen, it will recursively change ownership.
Also checks and updates the bluetooth service if needed.
"""

import filecmp
import logging
import subprocess
from pathlib import Path

logger = logging.getLogger(__name__)
USER = "pollen"


def check_and_fix_venvs_ownership(
    venvs_path: str = "/venvs", custom_logger: logging.Logger | None = None
) -> None:
    """For wireless units, check if files under venvs_path are owned by user pollen and fix if needed.

    Args:
        venvs_path: Path to the virtual environments directory (default: /venvs)
        custom_logger: Optional logger to use instead of the module logger

    """
    import pwd

    try:
        # Get pollen user's UID
        pollen_uid = pwd.getpwnam(USER).pw_uid
    except KeyError:
        print(f"User '{USER}' does not exist on this system")
        return

    venvs_dir = Path(venvs_path)

    if not venvs_dir.exists():
        print(f"Directory {venvs_path} does not exist")
        return

    if not venvs_dir.is_dir():
        print(f"{venvs_path} exists but is not a directory")
        return

    # Check if any files are not owned by pollen
    needs_fix = False
    try:
        for item in venvs_dir.rglob("*"):
            try:
                if item.stat().st_uid != pollen_uid:
                    needs_fix = True
                    print(f"Found file not owned by {USER}: {item}")
                    break
            except (PermissionError, OSError) as e:
                print(f"Cannot check ownership of {item}: {e}")
    except (PermissionError, OSError) as e:
        print(f"Cannot access {venvs_path}: {e}")
        return

    if needs_fix:
        print(f"Fixing ownership of {venvs_path} to {USER}:{USER}")
        try:
            # Run chown with sudo to fix ownership
            subprocess.run(
                ["sudo", "chown", f"{USER}:{USER}", "-R", venvs_path],
                check=True,
                capture_output=True,
                text=True,
            )
            print(f"Successfully fixed ownership of {venvs_path}")
        except subprocess.CalledProcessError as e:
            print(f"Failed to fix ownership: {e.stderr}")
        except Exception as e:
            print(f"Unexpected error while fixing ownership: {e}")
    else:
        print(f"All files under {venvs_path} are owned by {USER}")


def check_and_update_bluetooth_service() -> None:
    """Check if bluetooth service needs updating and update if different.

    Compares the source bluetooth_service.py with the installed version at
    /bluetooth/bluetooth_service.py. If they differ, copies the new version
    and restarts the bluetooth service. Also syncs the commands/ folder.
    """
    # This file: src/reachy_mini/utils/wireless_version/startup_check.py
    # Target:    src/reachy_mini/daemon/app/services/bluetooth/bluetooth_service.py
    # From parent: ../../daemon/app/services/bluetooth/bluetooth_service.py
    bluetooth_dir = (
        Path(__file__).parent
        / ".."
        / ".."
        / "daemon"
        / "app"
        / "services"
        / "bluetooth"
    )
    bluetooth_dir = bluetooth_dir.resolve()
    source = bluetooth_dir / "bluetooth_service.py"
    target = Path("/bluetooth/bluetooth_service.py")
    source_commands = bluetooth_dir / "commands"
    target_commands = Path("/bluetooth/commands")

    if not source.exists():
        print(f"Source bluetooth service not found at {source}")
        return

    needs_update = False
    needs_commands_update = False

    # Check if bluetooth_service.py needs update
    if not target.exists():
        print(f"Bluetooth service not installed at {target}, copying...")
        needs_update = True
    else:
        try:
            if not filecmp.cmp(str(source), str(target), shallow=False):
                print("Bluetooth service has changed, updating...")
                needs_update = True
        except Exception as e:
            print(f"Error comparing bluetooth service files: {e}")

    # Check if commands folder needs update
    if source_commands.exists():
        if not target_commands.exists():
            print("Commands folder not installed, copying...")
            needs_commands_update = True
        else:
            # Compare each command file
            for cmd_file in source_commands.glob("*.sh"):
                target_cmd = target_commands / cmd_file.name
                if not target_cmd.exists():
                    needs_commands_update = True
                    break
                try:
                    if not filecmp.cmp(str(cmd_file), str(target_cmd), shallow=False):
                        needs_commands_update = True
                        break
                except Exception:
                    needs_commands_update = True
                    break

    if not needs_update and not needs_commands_update:
        print("Bluetooth service and commands are up to date")
        return

    try:
        if needs_update:
            print(f"Copying {source} to {target}")
            subprocess.run(
                ["sudo", "cp", str(source), str(target)],
                check=True,
                capture_output=True,
                text=True,
            )
            print("Successfully copied bluetooth service")

        if needs_commands_update:
            print(f"Syncing commands folder to {target_commands}")
            subprocess.run(
                ["sudo", "mkdir", "-p", str(target_commands)],
                check=True,
                capture_output=True,
                text=True,
            )
            subprocess.run(
                ["sudo", "cp", "-r", f"{source_commands}/.", str(target_commands)],
                check=True,
                capture_output=True,
                text=True,
            )
            print("Successfully synced commands folder")

        # Restart the bluetooth service
        print("Restarting bluetooth service...")
        subprocess.run(
            ["sudo", "systemctl", "restart", "reachy-mini-bluetooth"],
            check=True,
            capture_output=True,
            text=True,
        )
        print("Successfully restarted bluetooth service")
    except subprocess.CalledProcessError as e:
        print(f"Failed to update bluetooth service: {e.stderr}")
    except Exception as e:
        print(f"Unexpected error while updating bluetooth service: {e}")


def check_and_update_wireless_launcher() -> None:
    """Check if wireless daemon service needs updating and update if different.

    Compares the source reachy-mini-daemon.service with the installed version.
    If they differ, copies the new version and reloads systemd.
    """
    source = (
        Path(__file__).parent
        / ".."
        / ".."
        / "daemon"
        / "app"
        / "services"
        / "wireless"
        / "reachy-mini-daemon.service"
    )
    source = source.resolve()
    target = Path("/etc/systemd/system/reachy-mini-daemon.service")

    if not source.exists():
        print(f"Source service file not found at {source}")
        return

    # Check if target exists
    if not target.exists():
        print(f"Wireless daemon service not installed at {target}")
        return

    # Compare files
    try:
        if filecmp.cmp(str(source), str(target), shallow=False):
            print("Wireless daemon service is up to date")
            return
        else:
            print("Wireless daemon service has changed, updating...")
    except Exception as e:
        print(f"Error comparing service files: {e}")
        return

    # Update service file
    try:
        print(f"Copying {source} to {target}")
        subprocess.run(
            ["sudo", "cp", str(source), str(target)],
            check=True,
            capture_output=True,
            text=True,
        )
        print("Successfully copied service file")

        # Reload systemd daemon
        print("Reloading systemd daemon...")
        subprocess.run(
            ["sudo", "systemctl", "daemon-reload"],
            check=True,
            capture_output=True,
            text=True,
        )
        print("Successfully reloaded systemd")
    except subprocess.CalledProcessError as e:
        print(f"Failed to update service: {e.stderr}")
    except Exception as e:
        print(f"Unexpected error while updating service: {e}")


def check_and_sync_apps_venv_sdk() -> None:
    """Check if apps_venv SDK matches daemon install source and sync if needed.

    Compares both version AND install source (PyPI vs git ref). If daemon was
    installed from a git ref, apps_venv will be synced to the same ref.

    """
    import json
    import shutil

    from .update_available import get_install_source

    # Get daemon install info
    try:
        daemon_info = get_install_source("reachy_mini")
    except Exception as e:
        print(f"Could not get daemon SDK info: {e}")
        return

    # Check apps_venv exists
    apps_venv_python = Path("/venvs/apps_venv/bin/python")
    if not apps_venv_python.exists():
        print("apps_venv not found, skipping SDK sync")
        return

    # Get apps_venv install info by reading metadata directly (avoid importing from apps_venv)
    try:
        result = subprocess.run(
            [
                str(apps_venv_python),
                "-c",
                "import json; from importlib.metadata import distribution, version; "
                "d = distribution('reachy_mini'); v = version('reachy_mini'); "
                "r = {'version': v, 'source': 'pypi'}; "
                "t = d.read_text('direct_url.json'); "
                "u = json.loads(t) if t else None; "
                "r.update({'source': 'editable'} if u and u.get('dir_info', {}).get('editable') else {}); "
                "r.update({'source': 'git', 'git_ref': u.get('vcs_info', {}).get('requested_revision', 'unknown')} "
                "if u and 'vcs_info' in u else {}); "
                "print(json.dumps(r))",
            ],
            capture_output=True,
            text=True,
            timeout=10,
        )
        if result.returncode != 0:
            print(f"Could not get apps_venv SDK info: {result.stderr}")
            return
        apps_info = json.loads(result.stdout.strip())
    except subprocess.TimeoutExpired:
        print("Timeout getting apps_venv SDK info")
        return
    except Exception as e:
        print(f"Error getting apps_venv SDK info: {e}")
        return

    print(f"Daemon: {daemon_info['version']} (source={daemon_info['source']}, ref={daemon_info.get('git_ref')})")
    print(f"Apps:   {apps_info['version']} (source={apps_info['source']}, ref={apps_info.get('git_ref')})")

    # Check if sync needed
    if daemon_info["source"] == "git":
        # Git install: sync if different ref
        needs_sync = apps_info.get("git_ref") != daemon_info.get("git_ref")
    else:
        # PyPI install: sync if different version
        needs_sync = apps_info["version"] != daemon_info["version"]

    if not needs_sync:
        print("Apps venv SDK is up to date")
        return

    # Build install command
    use_uv = shutil.which("uv") is not None
    if daemon_info["source"] == "git" and daemon_info.get("git_ref"):
        git_url = f"git+https://github.com/pollen-robotics/reachy_mini.git@{daemon_info['git_ref']}"
        pkg = f"reachy-mini[gstreamer] @ {git_url}"
        extra = ["--force-reinstall"]
        print(f"Syncing apps_venv to git ref: {daemon_info['git_ref']}")
    else:
        pkg = f"reachy-mini[gstreamer]=={daemon_info['version']}"
        extra = []
        print(f"Syncing apps_venv to version: {daemon_info['version']}")

    if use_uv:
        cmd = ["uv", "pip", "install", "--python", str(apps_venv_python), pkg] + extra
    else:
        cmd = [str(Path("/venvs/apps_venv/bin/pip")), "install", pkg] + extra

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        if result.returncode == 0:
            print("Successfully synced apps_venv SDK")
        else:
            print(f"Failed to sync apps_venv SDK: {result.stderr}")
    except subprocess.TimeoutExpired:
        print("Timeout syncing apps_venv SDK")
    except Exception as e:
        print(f"Error syncing apps_venv SDK: {e}")


def check_and_fix_restore_venv() -> None:
    """Check if restore venv has editable install and fix if needed.

    The restore partition at /restore/venvs should have a proper PyPI install,
    not an editable install. If an editable install is detected, reinstall
    from PyPI with a known good version.
    """
    restore_python = Path("/restore/venvs/mini_daemon/bin/python")

    if not restore_python.exists():
        print("Restore venv not found, skipping")
        return

    # Check if editable install
    try:
        result = subprocess.run(
            [str(restore_python), "-m", "pip", "show", "reachy-mini"],
            capture_output=True,
            text=True,
            timeout=30,
        )
    except subprocess.TimeoutExpired:
        print("Timeout checking restore venv")
        return
    except Exception as e:
        print(f"Error checking restore venv: {e}")
        return

    if "Editable project location" in result.stdout:
        print("Legacy editable install detected in restore venv, reinstalling...")
        try:
            subprocess.run(
                [str(restore_python), "-m", "pip", "install", "reachy-mini==1.2.8"],
                check=True,
                capture_output=True,
                text=True,
                timeout=300,
            )
            print("Successfully reinstalled reachy-mini in restore venv")
        except subprocess.CalledProcessError as e:
            print(f"Failed to reinstall in restore venv: {e.stderr}")
        except subprocess.TimeoutExpired:
            print("Timeout reinstalling in restore venv")
        except Exception as e:
            print(f"Error reinstalling in restore venv: {e}")
    else:
        print("Restore venv install is correct")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    check_and_fix_venvs_ownership()
    check_and_update_bluetooth_service()
    check_and_sync_apps_venv_sdk()
