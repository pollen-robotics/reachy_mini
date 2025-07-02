"""Reachy Mini Dashboard API."""

import threading
from importlib.metadata import entry_points

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

app = FastAPI()

app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

current_app = None
current_app_name = None
app_thread = None


def list_apps():
    """List all available Reachy Mini apps."""
    return list(entry_points(group="reachy_mini_apps"))


def start_app_by_name(name):
    """Start a Reachy Mini app by its name."""
    global current_app, current_app_name, app_thread
    apps = list_apps()
    for ep in apps:
        if ep.name == name:
            stop_app()
            AppClass = ep.load()
            current_app = AppClass()
            current_app_name = name
            app_thread = threading.Thread(target=current_app.wrapped_run)
            app_thread.start()
            break


def stop_app():
    """Stop the currently running Reachy Mini app."""
    global current_app, current_app_name, app_thread
    if current_app:
        current_app.stop()
    if app_thread and app_thread.is_alive():
        app_thread.join(timeout=10)
    current_app = None
    current_app_name = None
    app_thread = None


@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    """Render the main dashboard page with the list of apps."""
    apps = list_apps()
    return templates.TemplateResponse(
        "index.html", {"request": request, "apps": apps, "current": current_app_name}
    )


@app.post("/start/{name}")
async def start(name: str):
    """Start a Reachy Mini app by its name."""
    start_app_by_name(name)
    return RedirectResponse(url="/", status_code=303)


@app.post("/stop")
async def stop():
    """Stop the currently running Reachy Mini app."""
    stop_app()
    return RedirectResponse(url="/", status_code=303)


@app.get("/api/status")
async def status():
    """Get the status of the currently running Reachy Mini app."""
    apps = list_apps()
    return JSONResponse(
        {"current": current_app_name, "available_apps": [ep.name for ep in apps]}
    )
