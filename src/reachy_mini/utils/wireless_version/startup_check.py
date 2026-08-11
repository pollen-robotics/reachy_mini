"""Check and fix ownership of files under /venvs directory.

This module ensures that all files under /venvs are owned by the pollen user.
If any files are not owned by pollen, it will recursively change ownership.
Also checks and updates the bluetooth service if needed.
"""

import filecmp
import hashlib
import json
import logging
import os
import subprocess
import tempfile
from pathlib import Path

logger = logging.getLogger(__name__)
USER = "pollen"

# The stamp certifies that the *expensive* startup checks (full /venvs
# ownership scan, apps_venv SDK sync probe, restore-venv integrity) completed
# successfully for the exact set of installed reachy_mini distributions
# recorded in it. Any pip/uv (re)install rewrites the dist-info RECORD files,
# which changes the signature and invalidates the stamp. The stamp lives
# inside /venvs so that SOFTWARE_RESET.sh (rm -rf /venvs) and a re-flash
# destroy it. Any error anywhere in the fast path falls back to the full
# checks.
DAEMON_VENV = Path("/venvs/mini_daemon")
APPS_VENV = Path("/venvs/apps_venv")
RESTORE_VENV = Path("/restore/venvs/mini_daemon")
STAMP_PATH = DAEMON_VENV / ".startup_check_stamp.json"
STAMP_FORMAT = 1

# systemd unit self-healed on every boot (see check_and_update_gpio_shutdown_service).
GPIO_SHUTDOWN_UNIT_TARGET = Path("/etc/systemd/system/gpio-shutdown-daemon.service")


def check_and_fix_venvs_ownership(venvs_path: str = "/venvs") -> bool:
    """For wireless units, check if files under venvs_path are owned by user pollen and fix if needed.

    Args:
        venvs_path: Path to the virtual environments directory (default: /venvs)

    Returns:
        True when everything under venvs_path is (now) owned by USER.

    """
    import pwd

    try:
        # Get pollen user's UID
        pollen_uid = pwd.getpwnam(USER).pw_uid
    except KeyError:
        logger.error(f"User '{USER}' does not exist on this system")
        return False

    venvs_dir = Path(venvs_path)

    if not venvs_dir.exists():
        logger.error(f"Directory {venvs_path} does not exist")
        return False

    if not venvs_dir.is_dir():
        logger.error(f"{venvs_path} exists but is not a directory")
        return False

    # Check if any files are not owned by pollen
    needs_fix = False
    try:
        for item in venvs_dir.rglob("*"):
            try:
                if item.stat().st_uid != pollen_uid:
                    needs_fix = True
                    logger.warning(f"Found file not owned by {USER}: {item}")
                    break
            except (PermissionError, OSError) as e:
                logger.warning(f"Cannot check ownership of {item}: {e}")
    except (PermissionError, OSError) as e:
        logger.error(f"Cannot access {venvs_path}: {e}")
        return False

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
            return True
        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to fix ownership: {e.stderr}")
            return False
        except Exception as e:
            logger.error(f"Unexpected error while fixing ownership: {e}")
            return False
    else:
        logger.info(f"All files under {venvs_path} are owned by {USER}")
        return True


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
        logger.warning(f"Source bluetooth service not found at {source}")
        return

    needs_update = False
    needs_commands_update = False

    # Check if bluetooth_service.py needs update
    if not target.exists():
        logger.info(f"Bluetooth service not installed at {target}, copying...")
        needs_update = True
    else:
        try:
            if not filecmp.cmp(str(source), str(target), shallow=False):
                logger.info("Bluetooth service has changed, updating...")
                needs_update = True
        except Exception as e:
            logger.error(f"Error comparing bluetooth service files: {e}")

    # Check if commands folder needs update
    if source_commands.exists():
        if not target_commands.exists():
            logger.info("Commands folder not installed, copying...")
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
        logger.info("Bluetooth service and commands are up to date")
        return

    try:
        if needs_update:
            logger.info(f"Copying {source} to {target}")
            subprocess.run(
                ["sudo", "cp", str(source), str(target)],
                check=True,
                capture_output=True,
                text=True,
            )
            logger.info("Successfully copied bluetooth service")

        if needs_commands_update:
            logger.info(f"Syncing commands folder to {target_commands}")
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
            logger.info("Successfully synced commands folder")

        # Restart the bluetooth service
        logger.info("Restarting bluetooth service...")
        subprocess.run(
            ["sudo", "systemctl", "restart", "reachy-mini-bluetooth"],
            check=True,
            capture_output=True,
            text=True,
        )
        logger.info("Successfully restarted bluetooth service")
    except subprocess.CalledProcessError as e:
        logger.error(f"Failed to update bluetooth service: {e.stderr}")
    except Exception as e:
        logger.error(f"Unexpected error while updating bluetooth service: {e}")


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
        logger.warning(f"Source service file not found at {source}")
        return

    # Check if target exists
    if not target.exists():
        logger.warning(f"Wireless daemon service not installed at {target}")
        return

    # Compare files
    try:
        if filecmp.cmp(str(source), str(target), shallow=False):
            logger.info("Wireless daemon service is up to date")
            return
        else:
            logger.info("Wireless daemon service has changed, updating...")
    except Exception as e:
        logger.error(f"Error comparing service files: {e}")
        return

    # Update service file
    try:
        logger.info(f"Copying {source} to {target}")
        subprocess.run(
            ["sudo", "cp", str(source), str(target)],
            check=True,
            capture_output=True,
            text=True,
        )
        logger.info("Successfully copied service file")

        # Reload systemd daemon
        logger.info("Reloading systemd daemon...")
        subprocess.run(
            ["sudo", "systemctl", "daemon-reload"],
            check=True,
            capture_output=True,
            text=True,
        )
        logger.info("Successfully reloaded systemd")
    except subprocess.CalledProcessError as e:
        logger.error(f"Failed to update service: {e.stderr}")
    except Exception as e:
        logger.error(f"Unexpected error while updating service: {e}")


def _revert_gpio_unit(target: Path, previous: bytes | None) -> None:
    """Undo a half-applied unit update so the next boot retries it.

    check_and_update_gpio_shutdown_service returns early once the packaged and
    installed units match, so a new file left behind after a failed
    daemon-reload / restart / enable would mark the work done forever. systemd
    would keep the old unit, or on a fresh install keep no enabled unit at all,
    so the power switch stops triggering a graceful `shutdown -h now` (no
    kernel gpio-shutdown overlay backs it up, this daemon is the only thing
    listening) and every power-off becomes an abrupt cut. No later boot would
    fix it either, since the files still match.
    """
    try:
        if previous is None:
            subprocess.run(
                ["sudo", "rm", "-f", str(target)],
                check=True,
                capture_output=True,
                text=True,
            )
            logger.warning(
                f"Removed the half-installed {target}; the next boot will retry"
            )
            return

        with tempfile.NamedTemporaryFile(delete=False) as tmp:
            tmp.write(previous)
        try:
            subprocess.run(
                ["sudo", "cp", tmp.name, str(target)],
                check=True,
                capture_output=True,
                text=True,
            )
        finally:
            os.unlink(tmp.name)
        logger.warning(f"Restored the previous {target}; the next boot will retry")
    except Exception as e:
        logger.error(
            f"Could not revert {target} after a failed update: {e}. The next boot "
            "will find a matching unit and skip the sync; recover with "
            "`sudo systemctl daemon-reload && sudo systemctl enable --now "
            "gpio-shutdown-daemon`."
        )


def check_and_update_gpio_shutdown_service() -> None:
    """Check if the gpio-shutdown systemd unit needs updating and update if different.

    Compares the packaged gpio-shutdown-daemon.service with the installed
    version at /etc/systemd/system. Older images generated the unit at install
    time with an ExecStart pointing at wherever the SDK lived back then, so
    without this sync a stale unit keeps running old launcher code forever.
    On a difference the packaged unit is copied over and the service is
    restarted; a missing unit is installed and enabled.
    """
    # parents[2] is the reachy_mini package root (this file lives in
    # utils/wireless_version/).
    source = (
        Path(__file__).parents[2]
        / "daemon"
        / "app"
        / "services"
        / "gpio_shutdown"
        / "gpio-shutdown-daemon.service"
    ).resolve()
    target = GPIO_SHUTDOWN_UNIT_TARGET

    if not source.exists():
        logger.warning(f"Source gpio-shutdown service file not found at {source}")
        return

    target_missing = not target.exists()
    if not target_missing:
        try:
            if filecmp.cmp(str(source), str(target), shallow=False):
                logger.info("gpio-shutdown service is up to date")
                return
            logger.info("gpio-shutdown service has changed, updating...")
        except Exception as e:
            logger.error(f"Error comparing gpio-shutdown service files: {e}")
            return
    else:
        logger.info(f"gpio-shutdown service not installed at {target}, installing...")

    # Once the copy lands, file equality alone would tell the next boot the work
    # is done, so nothing may stay copied unless every follow-up step succeeded.
    previous_unit = None if target_missing else target.read_bytes()
    copied = False

    try:
        logger.info(f"Copying {source} to {target}")
        subprocess.run(
            ["sudo", "cp", str(source), str(target)],
            check=True,
            capture_output=True,
            text=True,
        )
        copied = True
        subprocess.run(
            ["sudo", "systemctl", "daemon-reload"],
            check=True,
            capture_output=True,
            text=True,
        )
        if target_missing:
            # Fresh install: enable so it also starts on subsequent boots.
            subprocess.run(
                ["sudo", "systemctl", "enable", "--now", "gpio-shutdown-daemon"],
                check=True,
                capture_output=True,
                text=True,
            )
        else:
            subprocess.run(
                ["sudo", "systemctl", "restart", "gpio-shutdown-daemon"],
                check=True,
                capture_output=True,
                text=True,
            )
        logger.info("Successfully updated gpio-shutdown service")
    except subprocess.CalledProcessError as e:
        logger.error(f"Failed to update gpio-shutdown service: {e.stderr}")
        if copied:
            _revert_gpio_unit(target, previous_unit)
    except Exception as e:
        logger.error(f"Unexpected error while updating gpio-shutdown service: {e}")
        if copied:
            _revert_gpio_unit(target, previous_unit)


def check_and_sync_apps_venv_sdk() -> bool:
    """Check if apps_venv SDK matches daemon install source and sync if needed.

    Compares both version AND install source (PyPI vs git ref). If daemon was
    installed from a git ref, apps_venv will be synced to the same ref.

    Returns:
        True when apps_venv is (now) in sync, or absent.

    """
    import json
    import os

    from .update_available import get_install_source
    from .utils import build_install_command

    # Get daemon install info
    try:
        daemon_info = get_install_source("reachy_mini")
    except Exception as e:
        logger.error(f"Could not get daemon SDK info: {e}")
        return False

    # Check apps_venv exists
    apps_venv_python = APPS_VENV / "bin/python"
    if not apps_venv_python.exists():
        logger.info("apps_venv not found, skipping SDK sync")
        return True

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
            logger.error(f"Could not get apps_venv SDK info: {result.stderr}")
            return False
        apps_info = json.loads(result.stdout.strip())
    except subprocess.TimeoutExpired:
        logger.error("Timeout getting apps_venv SDK info")
        return False
    except Exception as e:
        logger.error(f"Error getting apps_venv SDK info: {e}")
        return False

    logger.info(
        f"Daemon: {daemon_info['version']} (source={daemon_info['source']}, ref={daemon_info.get('git_ref')})"
    )
    logger.info(
        f"Apps:   {apps_info['version']} (source={apps_info['source']}, ref={apps_info.get('git_ref')})"
    )

    # Check if sync needed
    if daemon_info["source"] == "git":
        # Git install: sync if different ref
        needs_sync = apps_info.get("git_ref") != daemon_info.get("git_ref")
    else:
        # PyPI install: sync if different version
        needs_sync = apps_info["version"] != daemon_info["version"]

    if not needs_sync:
        logger.info("Apps venv SDK is up to date")
        return True

    # Build install command
    cmd, extra_env = build_install_command(
        extras="",
        git_ref=daemon_info.get("git_ref") if daemon_info["source"] == "git" else None,
        version=daemon_info["version"] if daemon_info["source"] != "git" else None,
        python=apps_venv_python,
    )

    resolved_env = {**os.environ, **extra_env} if extra_env else None

    try:
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=300, env=resolved_env, cwd=Path.home())
        if result.returncode == 0:
            logger.info("Successfully synced apps_venv SDK")
            return True
        else:
            logger.error(f"Failed to sync apps_venv SDK: {result.stderr}")
            return False
    except subprocess.TimeoutExpired:
        logger.error("Timeout syncing apps_venv SDK")
        return False
    except Exception as e:
        logger.error(f"Error syncing apps_venv SDK: {e}")
        return False


def check_and_fix_restore_venv() -> bool:
    """Check if restore venv has editable install and fix if needed.

    The restore partition at /restore/venvs should have a proper PyPI install,
    not an editable install. If an editable install is detected, reinstall
    from PyPI with a known good version.

    Returns:
        True when the restore venv is (now) correct, or absent.

    """
    restore_python = RESTORE_VENV / "bin/python"

    if not restore_python.exists():
        logger.info("Restore venv not found, skipping")
        return True

    # Check if editable install
    try:
        result = subprocess.run(
            [str(restore_python), "-m", "pip", "show", "reachy-mini"],
            capture_output=True,
            text=True,
            timeout=30,
        )
    except subprocess.TimeoutExpired:
        logger.error("Timeout checking restore venv")
        return False
    except Exception as e:
        logger.error(f"Error checking restore venv: {e}")
        return False

    if "Editable project location" in result.stdout:
        logger.warning("Legacy editable install detected in restore venv, reinstalling...")
        try:
            subprocess.run(
                [str(restore_python), "-m", "pip", "install", "reachy-mini==1.2.8"],
                check=True,
                capture_output=True,
                text=True,
                timeout=300,
            )
            logger.info("Successfully reinstalled reachy-mini in restore venv")
            return True
        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to reinstall in restore venv: {e.stderr}")
            return False
        except subprocess.TimeoutExpired:
            logger.error("Timeout reinstalling in restore venv")
            return False
        except Exception as e:
            logger.error(f"Error reinstalling in restore venv: {e}")
            return False
    else:
        logger.info("Restore venv install is correct")
        return True


def clear_startup_stamp() -> None:
    """Remove the startup-check stamp so the next boot runs the full checks.

    Raises:
        OSError: If the stamp exists and cannot be removed. Callers decide how
            loud to be about it; a stamp we cannot delete points at a
            filesystem or ownership problem rather than at the stamp itself.

    """
    STAMP_PATH.unlink(missing_ok=True)


def _clear_startup_stamp_best_effort() -> None:
    """Clear the stamp on the boot path, where raising would abort daemon startup."""
    try:
        clear_startup_stamp()
    except OSError as e:
        logger.error(f"Could not remove startup stamp: {e}")


def _dist_record_sig(venv_root: Path) -> str | None:
    """Stat-based signature of the reachy_mini dist-info RECORD file(s) in a venv.

    Any pip/uv install, upgrade, downgrade or --force-reinstall rewrites the
    dist-info directory, so (path, mtime_ns, size) of RECORD changes on every
    (re)install. Returns None when no reachy_mini distribution is present.
    """
    records = sorted(
        venv_root.glob("lib/python*/site-packages/reachy_mini-*.dist-info/RECORD")
    )
    if not records:
        return None
    parts = []
    for record in records:
        st = record.stat()
        parts.append(f"{record}:{st.st_mtime_ns}:{st.st_size}")
    return ";".join(parts)


def _compute_stamp_signature() -> dict[str, object]:
    """Compute the signature the stamp is compared against (a few stat calls).

    Raises on anything unexpected: callers treat any exception as "no fast
    path" and run the full checks.
    """
    from .update_available import get_install_source

    daemon_info = get_install_source("reachy_mini")
    if daemon_info.get("source") == "editable":
        # Editable code can change without any metadata change: never fast-path.
        raise RuntimeError("editable install detected, fast path disabled")

    daemon_record = _dist_record_sig(DAEMON_VENV)
    if daemon_record is None:
        raise RuntimeError(f"no reachy_mini dist-info found in {DAEMON_VENV}")

    return {
        "format": STAMP_FORMAT,
        # Invalidates the stamp whenever the check logic itself changes.
        "checks_hash": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "daemon": daemon_info,
        "daemon_record": daemon_record,
        "apps_record": _dist_record_sig(APPS_VENV),
        "restore_record": _dist_record_sig(RESTORE_VENV),
    }


def _quick_ownership_ok() -> bool:
    """Cheap, bounded ownership probe used on the fast path (no full rglob)."""
    import pwd

    uid = pwd.getpwnam(USER).pw_uid
    probes = [
        Path("/venvs"),
        DAEMON_VENV,
        DAEMON_VENV / "bin/python",
        APPS_VENV,
        STAMP_PATH,
    ]
    for probe in probes:
        if probe.exists() and probe.stat().st_uid != uid:
            logger.warning(f"Fast path aborted: {probe} not owned by {USER}")
            return False
    return True


def _startup_stamp_valid() -> bool:
    """Check that the stored stamp matches the freshly computed signature."""
    stored = json.loads(STAMP_PATH.read_text())
    if stored != _compute_stamp_signature():
        logger.warning("Startup stamp mismatch (install state changed); running full startup checks")
        return False
    return _quick_ownership_ok()


def _write_startup_stamp() -> None:
    """Atomically write the stamp after all expensive checks succeeded."""
    signature = _compute_stamp_signature()
    tmp = STAMP_PATH.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(signature, sort_keys=True))
    os.replace(tmp, STAMP_PATH)
    logger.info(f"Startup stamp written to {STAMP_PATH}")


def run_wireless_startup_checks() -> None:
    """Run wireless startup checks, skipping the expensive ones when nothing changed.

    Cheap checks (bluetooth service files, wireless and gpio-shutdown systemd
    units) always run: they guard files outside /venvs that a venv-keyed stamp
    cannot see. The expensive ones (full /venvs ownership scan, apps_venv SDK
    sync, restore-venv integrity) are skipped only when a stamp written by a
    previous fully-successful run matches the current install signature.
    Any error reading or matching the stamp falls back to the full checks.
    """
    try:
        fast_path = _startup_stamp_valid()
    except Exception as e:
        logger.warning(f"Startup stamp not usable ({e}); running full startup checks")
        fast_path = False

    if fast_path:
        logger.info("Startup stamp valid, skipping expensive startup checks")
        check_and_update_bluetooth_service()
        check_and_update_wireless_launcher()
        check_and_update_gpio_shutdown_service()
        return

    all_ok = check_and_fix_venvs_ownership()
    check_and_update_bluetooth_service()
    check_and_update_wireless_launcher()
    check_and_update_gpio_shutdown_service()
    all_ok = check_and_sync_apps_venv_sdk() and all_ok
    all_ok = check_and_fix_restore_venv() and all_ok

    if not all_ok:
        logger.warning("Some startup checks did not succeed; not writing startup stamp")
        _clear_startup_stamp_best_effort()
        return

    try:
        _write_startup_stamp()
    except Exception as e:
        # Never fatal: worst case the full checks simply run again next boot.
        # Drop any stale stamp so what's on disk is always fresh or absent.
        logger.error(f"Could not write startup stamp: {e}")
        _clear_startup_stamp_best_effort()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    run_wireless_startup_checks()
