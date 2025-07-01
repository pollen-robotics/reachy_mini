import asyncio
import json
import os
import shutil
import sys
import threading
import uuid
from datetime import datetime
from importlib.metadata import entry_points
from pathlib import Path
from typing import Dict, List, Optional

from app_install import (
    active_installations,
    broadcast_installation_status,
    connected_clients,
    install_app_async,
    installation_history,
    remove_app_async,
    update_app_async,
)
from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from utils import get_platform_info
from venv_app import VenvAppManager

app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# App state
current_app = None
current_app_name = None
app_thread = None
app_process = None

# Directories
DASHBOARD_DIR = Path(__file__).parent.absolute()
APPS_DIR = DASHBOARD_DIR / "installed_apps"
APPS_DIR.mkdir(exist_ok=True)

app_manager = VenvAppManager(APPS_DIR)

# Mount static files and templates
static_dir = DASHBOARD_DIR / "static"
templates_dir = DASHBOARD_DIR / "templates"

if static_dir.exists():
    app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")

if templates_dir.exists():
    templates = Jinja2Templates(directory=str(templates_dir))
else:
    templates = None


def list_apps():
    """List available apps from entry points and installed venv apps"""
    entry_point_apps = list(entry_points(group="reachy_mini_apps"))
    venv_apps_detailed = app_manager.list_installed_apps()
    venv_app_names = [app["name"] for app in venv_apps_detailed]

    all_apps = [ep.name for ep in entry_point_apps] + venv_app_names
    return list(set(all_apps))


def get_detailed_apps_info():
    """Get detailed information about all apps"""
    entry_point_apps = [
        {"name": ep.name, "type": "entry_point"}
        for ep in entry_points(group="reachy_mini_apps")
    ]
    venv_apps = app_manager.list_installed_apps()
    for app in venv_apps:
        app["type"] = "venv"

    return {
        "entry_point_apps": entry_point_apps,
        "venv_apps": venv_apps,
        "all_apps": [app["name"] for app in entry_point_apps + venv_apps],
    }


def start_app_by_name(name):
    global current_app, current_app_name, app_thread, app_process

    stop_app()
    current_app_name = name

    # Check if it's a venv app
    venv_app_names = [app["name"] for app in app_manager.list_installed_apps()]
    if name in venv_app_names:
        try:
            app_process = app_manager.run_app_in_venv(name)
            return
        except Exception as e:
            print(f"Failed to start venv app {name}: {e}")
            current_app_name = None
            raise

    # Try entry point apps
    apps = entry_points(group="reachy_mini_apps")
    for ep in apps:
        if ep.name == name:
            try:
                AppClass = ep.load()
                current_app = AppClass()

                if hasattr(current_app, "wrapped_run"):
                    app_thread = threading.Thread(target=current_app.wrapped_run)
                elif hasattr(current_app, "run"):
                    app_thread = threading.Thread(target=current_app.run)
                else:
                    raise Exception(f"App {name} has no run() or wrapped_run() method")
                app_thread.start()
                return
            except Exception as e:
                print(f"Failed to start entry point app {name}: {e}")
                current_app_name = None
                raise

    raise Exception(f"App '{name}' not found")


def stop_app():
    global current_app, current_app_name, app_thread, app_process

    # Stop venv app process
    if app_process:
        try:
            app_process.terminate()
            app_process.wait(timeout=5)
        except Exception:
            try:
                app_process.kill()
            except Exception:
                pass
        app_process = None

    # Stop entry point app
    if current_app:
        try:
            if hasattr(current_app, "stop"):
                current_app.stop()
        except Exception:
            pass
        current_app = None

    if app_thread and app_thread.is_alive():
        app_thread.join(timeout=5)
    app_thread = None

    current_app_name = None


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket endpoint for real-time installation updates"""
    await websocket.accept()
    connected_clients.append(websocket)

    try:
        # Send current active installations on connect
        if active_installations:
            for installation_id, status in active_installations.items():
                await websocket.send_text(
                    json.dumps(
                        {
                            "type": "installation_update",
                            "installation_id": installation_id,
                            "status": status,
                        }
                    )
                )

        # Keep connection alive with ping/pong
        while True:
            try:
                await asyncio.wait_for(websocket.receive_text(), timeout=30)
            except asyncio.TimeoutError:
                # Send a ping to keep connection alive
                await websocket.send_text(json.dumps({"type": "ping"}))

    except WebSocketDisconnect:
        if websocket in connected_clients:
            connected_clients.remove(websocket)
    except Exception:
        if websocket in connected_clients:
            connected_clients.remove(websocket)


@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    if not templates:
        return HTMLResponse(
            content="<h1>Dashboard templates not found</h1>", status_code=500
        )

    apps = list_apps()
    return templates.TemplateResponse(
        "index.html",
        {
            "request": request,
            "apps": apps,
            "current": current_app_name,
            "active_installations": active_installations,
            "venv_apps": app_manager.list_installed_apps(),
        },
    )


@app.post("/start/{name}")
async def start(name: str):
    try:
        start_app_by_name(name)
        return JSONResponse(
            content={
                "message": f"App '{name}' started successfully",
                "status": "running",
            }
        )
    except Exception as e:
        return JSONResponse(
            status_code=500, content={"detail": f"Failed to start app: {str(e)}"}
        )


@app.post("/stop")
async def stop():
    stop_app()
    return JSONResponse(
        content={"message": "App stopped successfully", "status": "stopped"}
    )


@app.get("/api/status")
async def status():
    """Get current status - limit data to reduce spam"""
    return JSONResponse(
        {
            "current": current_app_name,
            "available_apps": list_apps(),
            "active_installations_count": len(active_installations),
            "has_active_installations": len(active_installations) > 0,
            # Only send full details if there are active installations
            "active_installations": active_installations
            if active_installations
            else {},
        }
    )


@app.get("/api/status/full")
async def status_full():
    """Get full status information"""
    detailed_info = get_detailed_apps_info()
    return JSONResponse(
        {
            "current": current_app_name,
            "available_apps": detailed_info["all_apps"],
            "venv_apps": [app["name"] for app in detailed_info["venv_apps"]],
            "venv_apps_detailed": detailed_info["venv_apps"],
            "entry_point_apps": [
                app["name"] for app in detailed_info["entry_point_apps"]
            ],
            "active_installations": active_installations,
            "installation_history": installation_history[-10:],
        }
    )


@app.post("/api/install")
async def install_app(request: Request):
    """Handle app installation requests"""
    try:
        data = await request.json()
        app_url = data.get("url")
        app_name = data.get("name")

        if not app_url:
            return JSONResponse(
                status_code=400, content={"detail": "App URL is required"}
            )

        if not (app_url.startswith("http://") or app_url.startswith("https://")):
            return JSONResponse(
                status_code=400, content={"detail": "Invalid URL format"}
            )

        if (
            "github.com" in app_url
            or "gitlab.com" in app_url
            or app_url.endswith(".git")
        ):
            if not shutil.which("git"):
                return JSONResponse(
                    status_code=400, content={"detail": "Git is not installed"}
                )

        if not app_name:
            from urllib.parse import urlparse

            parsed_url = urlparse(app_url)
            app_name = (
                os.path.basename(parsed_url.path)
                .replace(".git", "")
                .replace(".zip", "")
            )
            if not app_name:
                app_name = parsed_url.netloc.replace(".", "_")

        # Sanitize app name
        app_name = "".join(c for c in app_name if c.isalnum() or c in "-_")

        venv_app_names = [app["name"] for app in app_manager.list_installed_apps()]
        if app_name in venv_app_names:
            return JSONResponse(
                status_code=400, content={"detail": f"App '{app_name}' already exists"}
            )

        installation_id = str(uuid.uuid4())

        # Start installation
        asyncio.create_task(
            install_app_async(
                installation_id,
                app_url,
                app_name,
                app_manager,
                DASHBOARD_DIR,
                current_app_name,
                stop_app,
            )
        )

        return JSONResponse(
            content={
                "message": "Installation started",
                "installation_id": installation_id,
                "app_name": app_name,
                "app_url": app_url,
                "status": "started",
            }
        )

    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"detail": f"Failed to start installation: {str(e)}"},
        )


@app.post("/api/apps/{app_name}/update")
async def update_app(app_name: str, request: Request):
    """Update a venv app to the latest version"""
    try:
        data = await request.json() if await request.body() else {}
        app_url = data.get("url")

        venv_app_names = [app["name"] for app in app_manager.list_installed_apps()]
        if app_name not in venv_app_names:
            return JSONResponse(
                status_code=404, content={"detail": f"App '{app_name}' not found"}
            )

        update_id = str(uuid.uuid4())
        asyncio.create_task(update_app_async(update_id, app_name, app_url, app_manager))

        return JSONResponse(
            content={
                "message": f"Update started for '{app_name}'",
                "update_id": update_id,
                "app_name": app_name,
                "status": "started",
            }
        )

    except Exception as e:
        return JSONResponse(
            status_code=500, content={"detail": f"Failed to start update: {str(e)}"}
        )


@app.delete("/api/apps/{app_name}")
async def remove_app(app_name: str):
    """Remove a venv app"""
    try:
        venv_app_names = [app["name"] for app in app_manager.list_installed_apps()]
        if app_name not in venv_app_names:
            return JSONResponse(
                status_code=404, content={"detail": f"App '{app_name}' not found"}
            )

        removal_id = str(uuid.uuid4())
        asyncio.create_task(
            remove_app_async(
                removal_id, app_name, app_manager, current_app_name, stop_app
            )
        )

        return JSONResponse(
            content={
                "message": f"Removal started for '{app_name}'",
                "removal_id": removal_id,
                "app_name": app_name,
                "status": "started",
            }
        )

    except Exception as e:
        return JSONResponse(
            status_code=500, content={"detail": f"Failed to start removal: {str(e)}"}
        )


@app.get("/api/platform")
async def platform_info():
    """Get platform information"""
    info = get_platform_info()
    info.update(
        {
            "apps_directory": str(APPS_DIR.absolute()),
            "python_executable": sys.executable,
            "git_available": shutil.which("git") is not None,
            "pip_available": shutil.which("pip") is not None,
        }
    )
    return JSONResponse(info)


@app.get("/api/installations")
async def get_installations():
    """Get installation history and active installations"""
    return JSONResponse(
        {
            "active_installations": active_installations,
            "installation_history": installation_history,
            "platform": get_platform_info(),
        }
    )


# External endpoint for installation
# @app.post("/api/install/external")
# async def install_app_external(request: Request):
#     """External endpoint to trigger app installation"""
#     return await install_app(request)


if __name__ == "__main__":
    import uvicorn

    platform_info = get_platform_info()

    print("🚀 Starting Robot Dashboard Server...")
    print(f"🖥️  Platform: {platform_info['system']} {platform_info['release']}")
    print(f"🐍 Python: {platform_info['python_version'].split()[0]}")
    print(f"📁 Apps Directory: {APPS_DIR.absolute()}")
    print(f"🔧 Git Available: {'✅' if shutil.which('git') else '❌'}")
    print("📡 CORS enabled for cross-origin requests")
    print("🔧 Installation endpoints ready")
    print("📊 Real-time updates via WebSocket at /ws")
    print("-" * 60)

    uvicorn.run(app, host="0.0.0.0", port=8000)
