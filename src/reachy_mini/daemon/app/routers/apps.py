"""Apps router for apps management."""

from fastapi import APIRouter, Depends, HTTPException

from reachy_mini.apps import AppInfo, SourceKind
from reachy_mini.apps.manager import AppManager, AppStatus
from reachy_mini.daemon.app.dependencies import get_app_manager

router = APIRouter(
    prefix="/apps",
)


@router.get("/list-available/{source_kind}")
async def list_available_apps(
    source_kind: SourceKind,
    app_manager: "AppManager" = Depends(get_app_manager),
) -> list[AppInfo]:
    """List available apps (including not installed)."""
    return await app_manager.list_available_apps(source_kind)


@router.get("/list-available")
async def list_all_available_apps(
    app_manager: "AppManager" = Depends(get_app_manager),
) -> list[AppInfo]:
    """List all available apps (including not installed)."""
    return await app_manager.list_all_available_apps()


@router.post("/install")
async def install_app(
    app_info: AppInfo,
    app_manager: "AppManager" = Depends(get_app_manager),
):
    """Install a new app by its info."""
    return await app_manager.install_new_app(app_info)


@router.post("/remove/{app_name}")
async def remove_app(
    app_name: str,
    app_manager: "AppManager" = Depends(get_app_manager),
):
    """Remove an installed app by name."""
    return await app_manager.remove_app(app_name)


@router.post("/start/{app_name}")
async def start_app(
    app_name: str,
    app_manager: "AppManager" = Depends(get_app_manager),
) -> AppStatus:
    """Start an installed app by name."""
    try:
        return await app_manager.start_app(app_name)
    except RuntimeError as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.post("/stop-current-app")
async def stop_current_app(app_manager: "AppManager" = Depends(get_app_manager)):
    """Stop the currently running app."""
    try:
        return await app_manager.stop_current_app()
    except RuntimeError as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.post("/restart-current-app")
async def restart_app(
    app_manager: "AppManager" = Depends(get_app_manager),
) -> AppStatus:
    """Restart the currently running app."""
    try:
        return await app_manager.restart_current_app()
    except RuntimeError as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.get("/current-app-status")
async def current_app_status(
    app_manager: "AppManager" = Depends(get_app_manager),
) -> AppStatus | None:
    """Get the status of the currently running app, if any."""
    return await app_manager.current_app_status()
