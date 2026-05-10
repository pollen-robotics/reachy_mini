"""Autostart configuration router.

Exposes a small REST surface for the dashboard to read/write the app
autostart configuration and view related service status.

Add to the daemon app:
    from .routers import autostart
    app.include_router(autostart.router)
"""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path
from typing import Any

from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel, Field, field_validator

CONFIG_PATH = Path("/etc/reachy-mini/autostart.json")
APPS_VENV_PIP = "/venvs/apps_venv/bin/pip"
APPS_VENV_PYTHON = "/venvs/apps_venv/bin/python"

DEFAULT_CONFIG: dict[str, Any] = {
    "app_autostart_enabled": False,
    "app_module": None,
    "app_args": [],
}


class AutostartConfig(BaseModel):
    """Body for POST /api/autostart/config."""

    app_autostart_enabled: bool
    app_module: str | None = None
    app_args: list[str] = Field(default_factory=list)

    @field_validator("app_module")
    @classmethod
    def _validate_module(cls, v: str | None) -> str | None:
        if v is None or v == "":
            return None
        # Defensive: allow only Python module path characters.
        # No shell metachars, slashes, etc. — even though the launcher uses
        # exec rather than shell, keeping this tight is a useful guardrail.
        allowed = set(
            "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_."
        )
        if not all(ch in allowed for ch in v):
            raise ValueError(
                "app_module must contain only [A-Za-z0-9_.] (Python module path)"
            )
        return v


router = APIRouter(prefix="/api/autostart", tags=["autostart"])


def _read_config() -> dict[str, Any]:
    if not CONFIG_PATH.exists():
        return DEFAULT_CONFIG.copy()
    try:
        data = json.loads(CONFIG_PATH.read_text())
    except json.JSONDecodeError as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Autostart config is malformed: {e}",
        )
    # Merge over defaults so missing keys are filled in
    merged = DEFAULT_CONFIG.copy()
    merged.update(data)
    return merged


def _write_config(cfg: dict[str, Any]) -> None:
    CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
    tmp = CONFIG_PATH.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(cfg, indent=2) + "\n")
    tmp.replace(CONFIG_PATH)


def _systemctl(*args: str) -> tuple[int, str, str]:
    """Run systemctl with the given args; never raises."""
    if not shutil.which("systemctl"):
        return 1, "", "systemctl not found"
    try:
        proc = subprocess.run(
            ["systemctl", *args], capture_output=True, text=True, timeout=5
        )
        return proc.returncode, proc.stdout.strip(), proc.stderr.strip()
    except subprocess.TimeoutExpired:
        return 1, "", "systemctl timed out"


# ─── endpoints ────────────────────────────────────────────────────────────

@router.get("/config")
def get_config() -> dict[str, Any]:
    """Return the current autostart configuration."""
    return _read_config()


@router.post("/config")
def set_config(cfg: AutostartConfig) -> dict[str, Any]:
    """Write the autostart configuration. Takes effect on next boot."""
    new = cfg.model_dump()
    _write_config(new)
    return new


@router.get("/installed_apps")
def list_installed_apps() -> dict[str, Any]:
    """Best-effort enumeration of likely-Reachy-app packages in apps_venv.

    Heuristic only: returns packages whose name contains 'reachy' or that
    declare a dependency on `reachy_mini`. Users can also free-text any
    Python module path via the dashboard.
    """
    if not Path(APPS_VENV_PIP).exists():
        return {"apps": [], "error": f"{APPS_VENV_PIP} not found"}
    try:
        proc = subprocess.run(
            [APPS_VENV_PIP, "list", "--format=json"],
            capture_output=True,
            text=True,
            timeout=20,
        )
    except subprocess.TimeoutExpired:
        return {"apps": [], "error": "pip list timed out"}

    if proc.returncode != 0:
        return {"apps": [], "error": proc.stderr.strip()[:200]}

    try:
        all_pkgs = json.loads(proc.stdout)
    except json.JSONDecodeError:
        return {"apps": [], "error": "could not parse pip output"}

    apps: list[dict[str, str]] = []
    for p in all_pkgs:
        name = p.get("name", "")
        version = p.get("version", "")
        lname = name.lower()
        if "reachy" not in lname and "robot" not in lname:
            continue
        # Skip the SDK itself
        if lname in ("reachy-mini", "reachy_mini"):
            continue
        # Best-guess module: package name with - → _
        guessed_module = name.replace("-", "_")
        apps.append(
            {
                "name": name,
                "version": version,
                "suggested_module": f"{guessed_module}.main",
            }
        )
    return {"apps": apps}


@router.get("/service_status")
def service_status() -> dict[str, Any]:
    """Active/enabled state of reachy-app-autostart.service."""
    rc_active, active, _ = _systemctl("is-active", "reachy-app-autostart.service")
    rc_enabled, enabled, _ = _systemctl(
        "is-enabled", "reachy-app-autostart.service"
    )
    return {
        "active": active,
        "active_ok": rc_active == 0,
        "enabled": enabled,
        "enabled_ok": rc_enabled == 0,
    }


@router.get("/daemon_autostart_status")
def daemon_autostart_status() -> dict[str, Any]:
    """Whether reachy-mini-daemon.service is enabled at boot.

    We expose this read-only — disabling the daemon is documented as an
    SSH operation in the dashboard UI to avoid the footgun of disabling
    the very service that serves the dashboard.
    """
    rc, enabled, _ = _systemctl("is-enabled", "reachy-mini-daemon.service")
    return {"enabled": enabled, "ok": rc == 0}
