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


def check_and_fix_venvs_ownership(venvs_path: str = "/venvs") -> None:
    """Check if files under venvs_path are owned by user pollen and fix if needed.

    Args:
        venvs_path: Path to the virtual environments directory (default: /venvs)

    """
    try:
        # Get pollen user's UID
        pollen_uid = pwd.getpwnam(USER).pw_uid
    except KeyError:
        logger.error(f"User '{USER}' does not exist on this system")
        return

    venvs_dir = Path(venvs_path)

    if not venvs_dir.exists():
        logger.warning(f"Directory {venvs_path} does not exist")
        return

    if not venvs_dir.is_dir():
        logger.error(f"{venvs_path} exists but is not a directory")
        return

    # Check if any files are not owned by pollen
    needs_fix = False
    try:
        for item in venvs_dir.rglob("*"):
            try:
                if item.stat().st_uid != pollen_uid:
                    needs_fix = True
                    logger.info(f"Found file not owned by {USER}: {item}")
                    break
            except (PermissionError, OSError) as e:
                logger.warning(f"Cannot check ownership of {item}: {e}")
    except (PermissionError, OSError) as e:
        logger.error(f"Cannot access {venvs_path}: {e}")
        return

    if needs_fix:
        logger.info(f"Fixing ownership of {venvs_path} to {USER}:{USER}")
        try:
            # Run chown with sudo to fix ownership
            subprocess.run(
                ["sudo", "chown", f"{USER}:{USER}", "-R", venvs_path],
                check=True,
                capture_output=True,
                text=True,
            )
            logger.info(f"Successfully fixed ownership of {venvs_path}")
        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to fix ownership: {e.stderr}")
        except Exception as e:
            logger.error(f"Unexpected error while fixing ownership: {e}")
    else:
        logger.info(f"All files under {venvs_path} are owned by {USER}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    check_and_fix_venvs_ownership()
