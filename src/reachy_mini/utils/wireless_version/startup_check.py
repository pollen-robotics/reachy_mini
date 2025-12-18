"""Check and fix ownership of files under /venvs directory.

This module ensures that all files under /venvs are owned by the pollen user.
If any files are not owned by pollen, it will recursively change ownership.
"""

import logging
import os
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
    log = custom_logger if custom_logger is not None else logger
    try:
        # Get pollen user's UID
        pollen_uid = pwd.getpwnam(USER).pw_uid
    except KeyError:
        log.error(f"User '{USER}' does not exist on this system")
        return

    venvs_dir = Path(venvs_path)

    if not venvs_dir.exists():
        log.warning(f"Directory {venvs_path} does not exist")
        return

    if not venvs_dir.is_dir():
        log.error(f"{venvs_path} exists but is not a directory")
        return

    # Check if any files are not owned by pollen
    needs_fix = False
    try:
        for item in venvs_dir.rglob("*"):
            try:
                if item.stat().st_uid != pollen_uid:
                    needs_fix = True
                    log.info(f"Found file not owned by {USER}: {item}")
                    break
            except (PermissionError, OSError) as e:
                log.warning(f"Cannot check ownership of {item}: {e}")
    except (PermissionError, OSError) as e:
        log.error(f"Cannot access {venvs_path}: {e}")
        return

    if needs_fix:
        log.info(f"Fixing ownership of {venvs_path} to {USER}:{USER}")
        try:
            # Run chown with sudo to fix ownership
            subprocess.run(
                ["sudo", "chown", f"{USER}:{USER}", "-R", venvs_path],
                check=True,
                capture_output=True,
                text=True,
            )
            log.info(f"Successfully fixed ownership of {venvs_path}")
        except subprocess.CalledProcessError as e:
            log.error(f"Failed to fix ownership: {e.stderr}")
        except Exception as e:
            log.error(f"Unexpected error while fixing ownership: {e}")
    else:
        log.info(f"All files under {venvs_path} are owned by {USER}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    check_and_fix_venvs_ownership()
