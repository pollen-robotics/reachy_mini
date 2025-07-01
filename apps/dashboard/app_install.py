import asyncio
import json
import shutil
import subprocess
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List

# Shared state
active_installations: Dict[str, dict] = {}
installation_history: List[dict] = []
connected_clients: List = []


async def broadcast_installation_status(installation_id: str, status: dict):
    """Broadcast installation status to all connected WebSocket clients"""
    if not connected_clients:
        return

    message = {
        "type": "installation_update",
        "installation_id": installation_id,
        "status": status,
    }

    disconnected_clients = []
    for client in connected_clients:
        try:
            await client.send_text(json.dumps(message))
        except Exception:
            disconnected_clients.append(client)

    for client in disconnected_clients:
        connected_clients.remove(client)


async def install_app_async(
    installation_id: str,
    app_url: str,
    app_name: str,
    app_manager,
    dashboard_dir: Path,
    current_app_name: str,
    stop_app_func,
):
    """Install app in virtual environment with progress updates"""
    try:
        # Update status: Starting
        status = {
            "stage": "starting",
            "progress": 0,
            "message": "Starting installation...",
            "app_name": app_name,
            "app_url": app_url,
        }
        active_installations[installation_id] = status
        await broadcast_installation_status(installation_id, status)
        print(f"Starting installation for {app_name}...")

        # Stage 1: Create virtual environment
        status.update(
            {
                "stage": "creating_venv",
                "progress": 20,
                "message": f"Creating virtual environment for {app_name}...",
            }
        )
        active_installations[installation_id] = status
        await broadcast_installation_status(installation_id, status)

        print("creating venv")
        venv_path = app_manager.create_venv(app_name)
        app_dir = venv_path.parent

        # Stage 2: Get pip path and verify it exists
        print(f"Virtual environment created at {venv_path}")
        pip_path = app_manager.get_venv_pip(app_name)
        if not pip_path.exists():
            raise Exception(f"Pip not found in virtual environment: {pip_path}")

        # Stage 3: Install app from URL
        print(f"Installing {app_name} from {app_url}...")
        status.update(
            {
                "stage": "installing",
                "progress": 40,
                "message": f"Installing {app_name} from repository...",
            }
        )
        active_installations[installation_id] = status
        await broadcast_installation_status(installation_id, status)

        # Upgrade pip first with fallback
        print("Upgrading pip...")
        try:
            run_subprocess([str(pip_path), "install", "--upgrade", "pip"], timeout=60)
        except Exception as e:
            print(f"Pip upgrade failed, trying alternative method: {e}")
            try:
                # Try with --user flag or without upgrade
                run_subprocess(
                    [str(pip_path), "install", "--upgrade", "pip", "--user"], timeout=60
                )
            except Exception as e2:
                print(f"Alternative pip upgrade also failed: {e2}")
                print("Continuing without pip upgrade...")

        # Direct pip install
        app_url = convert_hf_spaces_url(app_url)  # Convert HF Spaces URL if needed
        print(f"Installing {app_name} from {app_url}...")
        run_subprocess([str(pip_path), "install", f"git+{app_url}"])

        # # Stage 4: Install dependencies
        # status.update(
        #     {
        #         "stage": "dependencies",
        #         "progress": 70,
        #         "message": "Installing dependencies...",
        #     }
        # )
        # active_installations[installation_id] = status
        # await broadcast_installation_status(installation_id, status)

        # # Look for requirements.txt
        # install_requirements(app_dir, app_name, pip_path)

        # Stage 5: Complete
        status.update(
            {
                "stage": "complete",
                "progress": 100,
                "message": f"✅ {app_name} installed successfully!",
                "completed_at": datetime.now().isoformat(),
            }
        )
        active_installations[installation_id] = status
        await broadcast_installation_status(installation_id, status)

        # Save metadata
        metadata = {
            "installed_date": datetime.now().isoformat(),
            "source_url": app_url,
            "last_updated": datetime.now().isoformat(),
            "package_name": get_package_name_from_url(app_url, app_name),
        }
        app_manager.save_app_metadata(app_name, metadata)

        # Add to history
        installation_history.append(
            {
                "installation_id": installation_id,
                "app_name": app_name,
                "app_url": app_url,
                "status": "completed",
                "installed_at": datetime.now().isoformat(),
            }
        )

        # Clean up after delay
        await asyncio.sleep(5)
        if installation_id in active_installations:
            del active_installations[installation_id]

    except Exception as e:
        print(f"Installation failed for {app_name}: {e}")
        await handle_installation_error(installation_id, app_name, app_url, str(e))


async def install_from_git(
    app_url: str, app_name: str, app_dir: Path, pip_path: Path, dashboard_dir: Path
):
    """Install app from git repository"""
    git_url = app_url if app_url.endswith(".git") else f"{app_url}.git"

    clone_dir = app_dir / "src"
    clone_dir.mkdir(exist_ok=True)

    package_name = get_package_name_from_url(app_url, app_name)
    repo_path = clone_dir / package_name

    # Clone repository
    run_subprocess(["git", "clone", git_url, str(repo_path)], cwd=str(dashboard_dir))

    # Check installation method
    has_pyproject = (repo_path / "pyproject.toml").exists()
    has_setup_py = (repo_path / "setup.py").exists()

    if has_pyproject or has_setup_py:
        # Install as editable package
        try:
            run_subprocess([str(pip_path), "install", "-e", str(repo_path)])
        except Exception:
            # Fallback to regular install
            run_subprocess([str(pip_path), "install", str(repo_path)])
    else:
        # Install requirements if available
        req_file = repo_path / "requirements.txt"
        if req_file.exists():
            run_subprocess([str(pip_path), "install", "-r", str(req_file)])


def install_requirements(app_dir: Path, app_name: str, pip_path: Path):
    """Install requirements from various possible locations"""
    possible_req_paths = [
        app_dir / "requirements.txt",
        app_dir / "src" / "requirements.txt",
        app_dir / "src" / app_name / "requirements.txt",
        app_dir / "requirements" / "base.txt",
    ]

    for requirements_file in possible_req_paths:
        if requirements_file.exists():
            try:
                run_subprocess([str(pip_path), "install", "-r", str(requirements_file)])
                break
            except Exception:
                continue  # Try next requirements file


def convert_hf_spaces_url(app_url: str) -> str:
    """Convert HF Spaces static URL to git repository URL"""
    if ".static.hf.space" in app_url:
        # Extract the space name from static URL
        # https://pollen-robotics-reachy-mini-app-example.static.hf.space/index.html
        # -> pollen-robotics-reachy-mini-app-example
        domain_part = app_url.split(".static.hf.space")[0]
        if "://" in domain_part:
            space_name = domain_part.split("://")[1]  # Remove https://
        else:
            space_name = domain_part

        # Convert underscores to slashes and construct proper HF URL
        # pollen-robotics-reachy-mini-app-example -> pollen-robotics/reachy_mini_app_example
        if "-" in space_name:
            parts = space_name.split("-")
            # Find the split point (usually after organization name)
            # This is a heuristic - may need adjustment based on naming patterns
            org_name = parts[0] + "-" + parts[1]  # pollen-robotics
            app_name = "_".join(parts[2:])  # reachy_mini_app_example
            git_url = f"https://huggingface.co/spaces/{org_name}/{app_name}"
        else:
            git_url = f"https://huggingface.co/spaces/{space_name}"

        print(f"Converted HF Spaces URL: {app_url} -> {git_url}")
        return git_url

    return app_url


def get_package_name_from_url(app_url: str, app_name: str) -> str:
    """Extract package name from URL"""
    if "github.com" in app_url or "gitlab.com" in app_url:
        url_parts = app_url.rstrip("/").split("/")
        if len(url_parts) >= 2:
            return url_parts[-1].replace(".git", "")
    return app_name


def run_subprocess(cmd: List[str], cwd: str = None, timeout: int = 300):
    """Run subprocess and raise exception on failure"""
    print(f"Running command: {' '.join(cmd)}")
    if cwd:
        print(f"Working directory: {cwd}")

    try:
        result = subprocess.run(
            cmd, cwd=cwd, capture_output=True, text=True, timeout=timeout
        )

        print(f"Command completed with exit code: {result.returncode}")
        if result.stdout:
            print(f"STDOUT:\n{result.stdout}")
        if result.stderr:
            print(f"STDERR:\n{result.stderr}")

        if result.returncode != 0:
            error_msg = f"Exit code: {result.returncode}\nSTDOUT: {result.stdout}\nSTDERR: {result.stderr}"
            raise Exception(f"Command failed: {' '.join(cmd)}\n{error_msg}")

        return result

    except subprocess.TimeoutExpired:
        print(f"Command timed out after {timeout}s: {' '.join(cmd)}")
        raise Exception(f"Command timed out after {timeout}s: {' '.join(cmd)}")
    except Exception as e:
        print(f"Command execution failed: {e}")
        raise


async def handle_installation_error(
    installation_id: str, app_name: str, app_url: str, error: str
):
    """Handle installation errors"""
    error_status = {
        "stage": "error",
        "progress": 0,
        "message": f"❌ Installation failed: {error}",
        "error": error,
        "app_name": app_name,
        "app_url": app_url,
    }
    active_installations[installation_id] = error_status
    await broadcast_installation_status(installation_id, error_status)

    installation_history.append(
        {
            "installation_id": installation_id,
            "app_name": app_name,
            "app_url": app_url,
            "status": "failed",
            "error": error,
            "failed_at": datetime.now().isoformat(),
        }
    )

    # Keep error visible longer
    await asyncio.sleep(10)
    if installation_id in active_installations:
        del active_installations[installation_id]


async def update_app_async(
    installation_id: str, app_name: str, app_url: str, app_manager
):
    """Update an existing app"""
    try:
        metadata = app_manager.get_app_metadata(app_name)
        if not app_url:
            app_url = metadata.get("source_url")

        if not app_url:
            raise Exception("No source URL found for update.")

        status = {
            "stage": "starting",
            "progress": 0,
            "message": f"Starting update for {app_name}...",
            "app_name": app_name,
            "app_url": app_url,
            "operation": "update",
        }
        active_installations[installation_id] = status
        await broadcast_installation_status(installation_id, status)

        pip_path = app_manager.get_venv_pip(app_name)
        if not pip_path.exists():
            raise Exception(f"Pip not found: {pip_path}")

        # Update pip with fallback
        status.update(
            {"stage": "updating_pip", "progress": 30, "message": "Updating pip..."}
        )
        active_installations[installation_id] = status
        await broadcast_installation_status(installation_id, status)

        try:
            run_subprocess([str(pip_path), "install", "--upgrade", "pip"], timeout=60)
        except Exception as e:
            print(f"Pip upgrade failed during update: {e}")
            print("Continuing without pip upgrade...")

        # Update the app
        status.update(
            {
                "stage": "updating_app",
                "progress": 60,
                "message": f"Updating {app_name}...",
            }
        )
        active_installations[installation_id] = status
        await broadcast_installation_status(installation_id, status)

        app_dir = app_manager.apps_dir / app_name
        package_name = metadata.get("package_name", app_name)

        if "github.com" in app_url or "gitlab.com" in app_url:
            repo_dir = app_dir / "src" / package_name
            if repo_dir.exists() and (repo_dir / ".git").exists():
                # Pull latest changes
                run_subprocess(["git", "pull"], cwd=str(repo_dir))
                # Reinstall
                run_subprocess(
                    [str(pip_path), "install", "-e", str(repo_dir), "--upgrade"]
                )
            else:
                # Re-install from git
                git_url = app_url if app_url.endswith(".git") else f"{app_url}.git"
                run_subprocess(
                    [
                        str(pip_path),
                        "install",
                        "--upgrade",
                        "--force-reinstall",
                        f"git+{git_url}",
                    ]
                )
        else:
            run_subprocess([str(pip_path), "install", "--upgrade", app_url])

        # Complete
        status.update(
            {
                "stage": "complete",
                "progress": 100,
                "message": f"✅ {app_name} updated successfully!",
                "completed_at": datetime.now().isoformat(),
            }
        )
        active_installations[installation_id] = status
        await broadcast_installation_status(installation_id, status)

        # Update metadata
        metadata.update(
            {"last_updated": datetime.now().isoformat(), "source_url": app_url}
        )
        app_manager.save_app_metadata(app_name, metadata)

        installation_history.append(
            {
                "installation_id": installation_id,
                "app_name": app_name,
                "app_url": app_url,
                "status": "updated",
                "updated_at": datetime.now().isoformat(),
            }
        )

        await asyncio.sleep(5)
        if installation_id in active_installations:
            del active_installations[installation_id]

    except Exception as e:
        await handle_update_error(installation_id, app_name, app_url, str(e))


async def handle_update_error(
    installation_id: str, app_name: str, app_url: str, error: str
):
    """Handle update errors"""
    error_status = {
        "stage": "error",
        "progress": 0,
        "message": f"❌ Update failed: {error}",
        "error": error,
        "app_name": app_name,
        "app_url": app_url,
        "operation": "update",
    }
    active_installations[installation_id] = error_status
    await broadcast_installation_status(installation_id, error_status)

    installation_history.append(
        {
            "installation_id": installation_id,
            "app_name": app_name,
            "app_url": app_url,
            "status": "update_failed",
            "error": error,
            "failed_at": datetime.now().isoformat(),
        }
    )

    await asyncio.sleep(10)
    if installation_id in active_installations:
        del active_installations[installation_id]


async def remove_app_async(
    removal_id: str, app_name: str, app_manager, current_app_name: str, stop_app_func
):
    """Remove an app with progress updates"""
    try:
        status = {
            "stage": "starting",
            "progress": 0,
            "message": f"Starting removal of {app_name}...",
            "app_name": app_name,
            "operation": "remove",
        }
        active_installations[removal_id] = status
        await broadcast_installation_status(removal_id, status)

        # Stop app if running
        if current_app_name == app_name:
            status.update(
                {
                    "stage": "stopping",
                    "progress": 25,
                    "message": "Stopping running app...",
                }
            )
            active_installations[removal_id] = status
            await broadcast_installation_status(removal_id, status)
            stop_app_func()

        # Remove files
        status.update(
            {
                "stage": "removing_files",
                "progress": 75,
                "message": "Removing application files...",
            }
        )
        active_installations[removal_id] = status
        await broadcast_installation_status(removal_id, status)

        success = app_manager.remove_app_completely(app_name)
        if not success:
            raise Exception("Failed to remove app directory")

        # Complete
        status.update(
            {
                "stage": "complete",
                "progress": 100,
                "message": f"✅ {app_name} removed successfully!",
                "completed_at": datetime.now().isoformat(),
            }
        )
        active_installations[removal_id] = status
        await broadcast_installation_status(removal_id, status)

        installation_history.append(
            {
                "installation_id": removal_id,
                "app_name": app_name,
                "status": "removed",
                "removed_at": datetime.now().isoformat(),
            }
        )

        await asyncio.sleep(3)
        if removal_id in active_installations:
            del active_installations[removal_id]

    except Exception as e:
        await handle_removal_error(removal_id, app_name, str(e))


async def handle_removal_error(removal_id: str, app_name: str, error: str):
    """Handle removal errors"""
    error_status = {
        "stage": "error",
        "progress": 0,
        "message": f"❌ Removal failed: {error}",
        "error": error,
        "app_name": app_name,
        "operation": "remove",
    }
    active_installations[removal_id] = error_status
    await broadcast_installation_status(removal_id, error_status)

    installation_history.append(
        {
            "installation_id": removal_id,
            "app_name": app_name,
            "status": "removal_failed",
            "error": error,
            "failed_at": datetime.now().isoformat(),
        }
    )

    await asyncio.sleep(10)
    if removal_id in active_installations:
        del active_installations[removal_id]
