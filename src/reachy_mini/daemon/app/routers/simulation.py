"""Simulation status router for MuJoCo web simulation."""

import logging
from pathlib import Path

from fastapi import APIRouter
from pydantic import BaseModel

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/simulation", tags=["simulation"])


class SimulationStatus(BaseModel):
    """Simulation status response."""
    available: bool
    wasm_files_present: bool
    message: str


@router.get("/status", response_model=SimulationStatus)
async def get_simulation_status():
    """Check if simulation files are available."""
    # Check if WASM files exist
    app_dir = Path(__file__).parent.parent
    wasm_dir = app_dir / "wasm" / "dist"

    wasm_file = wasm_dir / "assets" / "mujoco_wasm-C1UIFeB5.wasm"
    js_file = wasm_dir / "assets" / "mujoco_wasm-QTWZXvSS.js"
    index_file = wasm_dir / "index.html"

    files_present = all([
        wasm_file.exists(),
        js_file.exists(),
        index_file.exists()
    ])

    if files_present:
        message = "Simulation ready - served directly from FastAPI at /simulation/"
    else:
        message = "Simulation files not found - please build WASM first"

    return SimulationStatus(
        available=files_present,
        wasm_files_present=files_present,
        message=message
    )