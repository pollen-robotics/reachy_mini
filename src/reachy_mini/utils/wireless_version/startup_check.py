"""Check and fix ownership of files under /venvs directory.

This module ensures that all files under /venvs are owned by the pollen user.
If any files are not owned by pollen, it will recursively change ownership.
Also checks and updates the bluetooth service if needed.
"""

import filecmp
import logging
import pwd
import subprocess
from pathlib import Path

logger = logging.getLogger(__name__)
USER = "pollen"


def check_and_fix_venvs_ownership(
    venvs_path: str = "/venvs", custom_logger: logging.Logger | None = None
) -> None:
    """Check if files under venvs_path are owned by user pollen and fix if needed.

    Args:
        venvs_path: Path to the virtual environments directory (default: /venvs)
        custom_logger: Optional logger to use instead of the module logger

    """
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


def check_and_update_bluetooth_service(
    custom_logger: logging.Logger | None = None,
) -> None:
    """Check if bluetooth service needs updating and update if different.

    Compares the source bluetooth_service.py with the installed version at
    /bluetooth/bluetooth_service.py. If they differ, copies the new version
    and restarts the bluetooth service.

    Args:
        custom_logger: Optional logger to use instead of the module logger

    """
    log = custom_logger if custom_logger else logger

    # This file: src/reachy_mini/utils/wireless_version/startup_check.py
    # Target:    src/reachy_mini/daemon/app/services/bluetooth/bluetooth_service.py
    # From parent: ../../daemon/app/services/bluetooth/bluetooth_service.py
    source = Path(__file__).parent / ".." / ".." / "daemon" / "app" / "services" / "bluetooth" / "bluetooth_service.py"
    source = source.resolve()
    target = Path("/bluetooth/bluetooth_service.py")

    if not source.exists():
        log.error(f"Source bluetooth service not found at {source}")
        return

    # Check if target exists
    if not target.exists():
        log.info(f"Bluetooth service not installed at {target}, copying...")
        needs_update = True
    else:
        # Compare files
        try:
            if filecmp.cmp(str(source), str(target), shallow=False):
                log.info("Bluetooth service is up to date")
                return
            else:
                log.info("Bluetooth service has changed, updating...")
                needs_update = True
        except Exception as e:
            log.error(f"Error comparing bluetooth service files: {e}")
            return

    if needs_update:
        try:
            # Copy the new version using sudo
            log.info(f"Copying {source} to {target}")
            subprocess.run(
                ["sudo", "cp", str(source), str(target)],
                check=True,
                capture_output=True,
                text=True,
            )
            log.info("Successfully copied bluetooth service")

            # Restart the bluetooth service
            log.info("Restarting bluetooth service...")
            subprocess.run(
                ["sudo", "systemctl", "restart", "reachy-mini-bluetooth"],
                check=True,
                capture_output=True,
                text=True,
            )
            log.info("Successfully restarted bluetooth service")
        except subprocess.CalledProcessError as e:
            log.error(f"Failed to update bluetooth service: {e.stderr}")
        except Exception as e:
            log.error(f"Unexpected error while updating bluetooth service: {e}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    check_and_fix_venvs_ownership()
    check_and_update_bluetooth_service()
