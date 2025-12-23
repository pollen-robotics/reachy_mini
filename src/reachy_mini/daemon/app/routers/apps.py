"""Apps router for apps management."""

from fastapi import (
    APIRouter,
    Depends,
    HTTPException,
    WebSocket,
)
from pydantic import BaseModel

from reachy_mini.apps import AppInfo, SourceKind
from reachy_mini.apps.manager import AppManager, AppStatus
from reachy_mini.daemon.app import bg_job_register
from reachy_mini.daemon.app.dependencies import get_app_manager

router = APIRouter(prefix="/apps")


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
) -> dict[str, str]:
    """Install a new app by its info (background, returns job_id)."""
    job_id = bg_job_register.run_command(
        "install", app_manager.install_new_app, app_info
    )
    return {"job_id": job_id}


@router.post("/remove/{app_name}")
async def remove_app(
    app_name: str,
    app_manager: "AppManager" = Depends(get_app_manager),
) -> dict[str, str]:
    """Remove an installed app by its name (background, returns job_id)."""
    job_id = bg_job_register.run_command("remove", app_manager.remove_app, app_name)
    return {"job_id": job_id}


@router.get("/job-status/{job_id}")
async def job_status(job_id: str) -> bg_job_register.JobInfo:
    """Get status/logs for a job."""
    try:
        return bg_job_register.get_info(job_id)
    except Exception as e:
        raise HTTPException(status_code=404, detail=str(e))


# WebSocket route for live job status/logs
@router.websocket("/ws/apps-manager/{job_id}")
async def ws_apps_manager(websocket: WebSocket, job_id: str) -> None:
    """WebSocket route to stream live job status/logs for a job, sending updates as soon as new logs are available."""
    await websocket.accept()
    await bg_job_register.ws_poll_info(websocket, job_id)
    await websocket.close()


@router.post("/start-app/{app_name}")
async def start_app(
    app_name: str,
    app_manager: "AppManager" = Depends(get_app_manager),
) -> AppStatus:
    """Start an app by its name."""
    try:
        return await app_manager.start_app(app_name)
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


@router.post("/stop-current-app")
async def stop_app(
    app_manager: "AppManager" = Depends(get_app_manager),
) -> None:
    """Stop the currently running app."""
    try:
        return await app_manager.stop_current_app()
    except RuntimeError as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.get("/current-app-status")
async def current_app_status(
    app_manager: "AppManager" = Depends(get_app_manager),
) -> AppStatus | None:
    """Get the status of the currently running app, if any."""
    return await app_manager.current_app_status()


class PrivateSpaceInstallRequest(BaseModel):
    """Request model for installing a private HuggingFace space."""

    space_id: str


@router.post("/install-private-space")
async def install_private_space(
    request: PrivateSpaceInstallRequest,
    app_manager: "AppManager" = Depends(get_app_manager),
) -> dict[str, str]:
    """Install a private HuggingFace space.

    Only available on wireless version.
    Requires HF token to be stored via /api/hf-auth/save-token first.
    """
    if not app_manager.wireless_version:
        raise HTTPException(
            status_code=403,
            detail="Private space installation only available on wireless version",
        )

    from reachy_mini.apps.sources import hf_auth

    # Check if token is available
    token = hf_auth.get_hf_token()
    if not token:
        raise HTTPException(
            status_code=401,
            detail="No HuggingFace token found. Please authenticate first.",
        )

    # Create AppInfo for the private space
    space_name = request.space_id.split("/")[-1]
    app_info = AppInfo(
        name=space_name,
        description=f"Private space: {request.space_id}",
        url=f"https://huggingface.co/spaces/{request.space_id}",
        source_kind=SourceKind.HF_SPACE,
        extra={
            "id": request.space_id,
            "private": True,
            "cardData": {
                "title": space_name,
                "short_description": f"Private space: {request.space_id}",
            },
        },
    )

    job_id = bg_job_register.run_command(
        "install", app_manager.install_new_app, app_info
    )
    return {"job_id": job_id}
