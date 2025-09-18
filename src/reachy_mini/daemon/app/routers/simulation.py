"""Simulation control router for MuJoCo web simulation."""

import asyncio
import logging
import os
import signal
import subprocess
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/simulation", tags=["simulation"])


class SimulationStartRequest(BaseModel):
    """Request model for starting simulation."""
    port: int = 3001


class SimulationStatus(BaseModel):
    """Simulation status response."""
    running: bool
    port: Optional[int] = None
    pid: Optional[int] = None


class SimulationManager:
    """Manages the MuJoCo simulation server process."""

    def __init__(self):
        self.process: Optional[subprocess.Popen] = None
        self.port: Optional[int] = None

    def is_running(self) -> bool:
        """Check if simulation process is running."""
        if self.process is None:
            return False
        return self.process.poll() is None

    async def start(self, port: int = 3001) -> bool:
        """Start the MuJoCo simulation server."""
        if self.is_running():
            logger.warning("Simulation server is already running")
            return True

        # Find the MuJoCo web directory
        app_dir = Path(__file__).parent.parent
        wasm_dir = app_dir / "wasm"

        # Use the built distribution in the dist directory
        dist_dir = wasm_dir / "dist"
        server_script = dist_dir / "server.js"

        if not server_script.exists():
            logger.error(f"MuJoCo server script not found at {server_script}")
            logger.error(f"Please run the build script in {wasm_dir} first")
            return False

        if not (dist_dir / "mujoco.wasm").exists():
            logger.error(f"MuJoCo WASM files not found in {dist_dir}")
            logger.error(f"Please run the build script in {wasm_dir} first")
            return False

        try:
            # Change to the directory containing the server script
            cwd = server_script.parent

            # Set environment variables
            env = os.environ.copy()
            env["PORT"] = str(port)

            # Start the Node.js server
            self.process = subprocess.Popen(
                ["node", "server.js"],
                cwd=str(cwd),
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                preexec_fn=os.setsid if os.name != 'nt' else None
            )

            self.port = port

            # Wait a moment to see if process starts successfully
            await asyncio.sleep(1)

            if not self.is_running():
                stdout, stderr = self.process.communicate()
                logger.error(f"Failed to start simulation server. stdout: {stdout.decode()}, stderr: {stderr.decode()}")
                return False

            logger.info(f"Started MuJoCo simulation server on port {port} with PID {self.process.pid}")
            return True

        except Exception as e:
            logger.error(f"Error starting simulation server: {e}")
            return False

    async def stop(self) -> bool:
        """Stop the MuJoCo simulation server."""
        if not self.is_running():
            logger.warning("Simulation server is not running")
            return True

        try:
            if os.name != 'nt':
                # Unix-like systems: kill process group
                os.killpg(os.getpgid(self.process.pid), signal.SIGTERM)
            else:
                # Windows
                self.process.terminate()

            # Wait for process to exit
            try:
                self.process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                # Force kill if it doesn't exit gracefully
                if os.name != 'nt':
                    os.killpg(os.getpgid(self.process.pid), signal.SIGKILL)
                else:
                    self.process.kill()
                self.process.wait()

            logger.info(f"Stopped MuJoCo simulation server (PID {self.process.pid})")
            self.process = None
            self.port = None
            return True

        except Exception as e:
            logger.error(f"Error stopping simulation server: {e}")
            return False


# Global simulation manager instance
simulation_manager = SimulationManager()


@router.get("/status", response_model=SimulationStatus)
async def get_simulation_status():
    """Get the current simulation status."""
    return SimulationStatus(
        running=simulation_manager.is_running(),
        port=simulation_manager.port,
        pid=simulation_manager.process.pid if simulation_manager.process else None
    )


@router.post("/start")
async def start_simulation(request: SimulationStartRequest):
    """Start the MuJoCo simulation server."""
    success = await simulation_manager.start(request.port)

    if not success:
        raise HTTPException(status_code=500, detail="Failed to start simulation server")

    return {
        "message": f"Simulation server started on port {request.port}",
        "port": request.port,
        "pid": simulation_manager.process.pid if simulation_manager.process else None
    }


@router.post("/stop")
async def stop_simulation():
    """Stop the MuJoCo simulation server."""
    success = await simulation_manager.stop()

    if not success:
        raise HTTPException(status_code=500, detail="Failed to stop simulation server")

    return {"message": "Simulation server stopped"}


@router.post("/restart")
async def restart_simulation(request: SimulationStartRequest):
    """Restart the MuJoCo simulation server."""
    # Stop first
    await simulation_manager.stop()

    # Wait a moment
    await asyncio.sleep(1)

    # Start again
    success = await simulation_manager.start(request.port)

    if not success:
        raise HTTPException(status_code=500, detail="Failed to restart simulation server")

    return {
        "message": f"Simulation server restarted on port {request.port}",
        "port": request.port,
        "pid": simulation_manager.process.pid if simulation_manager.process else None
    }