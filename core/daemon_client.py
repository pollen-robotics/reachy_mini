"""Daemon REST API client for Reachy Control Center.

Provides simple interface to communicate with the Reachy Mini daemon.
"""

import requests
import logging
import subprocess
import time
from typing import Any, Dict, List, Optional, Tuple
from pathlib import Path

logger = logging.getLogger(__name__)


class DaemonClient:
    """Client for interacting with Reachy Mini daemon REST API."""

    def __init__(self, base_url: str = "http://localhost:8100", timeout: float = 2.0):
        """Initialize daemon client.

        Args:
            base_url: Base URL of daemon API (default: http://localhost:8100)
            timeout: Request timeout in seconds (default: 2.0)
        """
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self._connected = False
        self._daemon_process: Optional[subprocess.Popen] = None

    def check_connection(self) -> bool:
        """Check if daemon is responding.

        Returns:
            True if daemon is reachable, False otherwise
        """
        try:
            response = requests.get(
                f"{self.base_url}/api/daemon/status",
                timeout=self.timeout
            )
            self._connected = response.status_code == 200
            return self._connected
        except Exception as e:
            logger.debug(f"Connection check failed: {e}")
            self._connected = False
            return False

    def get_status(self) -> Optional[Dict[str, Any]]:
        """Get daemon status.

        Returns:
            Status dictionary or None if request failed
        """
        try:
            response = requests.get(
                f"{self.base_url}/api/daemon/status",
                timeout=self.timeout
            )
            if response.status_code == 200:
                return response.json()
            return None
        except Exception as e:
            logger.error(f"Failed to get status: {e}")
            return None

    def get_state(self) -> Optional[Dict[str, Any]]:
        """Get current robot state (positions, etc).

        Returns:
            State dictionary or None if request failed
        """
        try:
            response = requests.get(
                f"{self.base_url}/api/state/full",
                timeout=self.timeout
            )
            if response.status_code == 200:
                return response.json()
            return None
        except Exception as e:
            logger.error(f"Failed to get state: {e}")
            return None

    def set_target(
        self,
        head_x: float = 0,
        head_y: float = 0,
        head_z: float = 0,
        head_yaw: float = 0,
        head_pitch: float = 0,
        head_roll: float = 0,
        left_antenna: float = 0,
        right_antenna: float = 0,
        duration: float = 0.5
    ) -> bool:
        """Send target position command to robot.

        Args:
            head_x: Head X position in mm
            head_y: Head Y position in mm
            head_z: Head Z position in mm
            head_yaw: Head yaw rotation in radians
            head_pitch: Head pitch rotation in radians
            head_roll: Head roll rotation in radians
            left_antenna: Left antenna position in radians
            right_antenna: Right antenna position in radians
            duration: Movement duration in seconds

        Returns:
            True if command succeeded, False otherwise
        """
        try:
            payload = {
                "target_head_pose": {
                    "x": head_x / 1000.0,  # Convert mm to meters
                    "y": head_y / 1000.0,
                    "z": head_z / 1000.0,
                    "yaw": head_yaw,
                    "pitch": head_pitch,
                    "roll": head_roll
                },
                "target_antennas": [left_antenna, right_antenna],
                "duration": duration
            }
            response = requests.post(
                f"{self.base_url}/api/move/set_target",
                json=payload,
                timeout=self.timeout
            )
            return response.status_code == 200
        except Exception as e:
            logger.error(f"Failed to set target: {e}")
            return False

    def play_move(self, dataset: str, move_name: str) -> Optional[str]:
        """Play a pre-recorded move.

        Args:
            dataset: Dataset name (e.g., "pollen-robotics/reachy-mini-dances-library")
            move_name: Move name (e.g., "wave")

        Returns:
            Move UUID if successful, None otherwise
        """
        try:
            response = requests.post(
                f"{self.base_url}/api/move/play/recorded-move-dataset/{dataset}/{move_name}",
                timeout=self.timeout
            )
            if response.status_code == 200:
                data = response.json()
                return data.get("uuid")
            return None
        except Exception as e:
            logger.error(f"Failed to play move: {e}")
            return None

    def stop_move(self, uuid: Optional[str] = None) -> bool:
        """Stop currently running move.

        Args:
            uuid: Move UUID to stop (optional)

        Returns:
            True if stop succeeded, False otherwise
        """
        try:
            payload = {}
            if uuid:
                payload["uuid"] = uuid

            response = requests.post(
                f"{self.base_url}/api/move/stop",
                json=payload,
                timeout=self.timeout
            )
            return response.status_code == 200
        except Exception as e:
            logger.error(f"Failed to stop move: {e}")
            return False

    def get_camera_frame(self) -> Optional[bytes]:
        """Get single camera frame as JPEG.

        Returns:
            JPEG image bytes or None if request failed
        """
        try:
            response = requests.get(
                f"{self.base_url}/api/camera/frame",
                timeout=self.timeout
            )
            if response.status_code == 200 and len(response.content) > 0:
                return response.content
            return None
        except Exception as e:
            logger.debug(f"Failed to get camera frame: {e}")
            return None

    def get_camera_stream_url(self) -> str:
        """Get URL for MJPEG camera stream.

        Returns:
            MJPEG stream URL
        """
        return f"{self.base_url}/api/camera/stream.mjpg"

    def list_available_moves(self, dataset: str) -> List[str]:
        """List available moves in a dataset.

        Args:
            dataset: Dataset name

        Returns:
            List of move names
        """
        # TODO: Add this endpoint to daemon if it doesn't exist
        # For now, return hardcoded lists
        if "dances" in dataset.lower():
            return [
                "side_to_side_sway", "jackson_square", "dizzy_spin",
                "stumble_and_recover", "chin_lead", "head_tilt_roll",
                "pendulum_swing", "side_glance_flick", "grid_snap",
                "simple_nod", "polyrhythm_combo", "interwoven_spirals",
                "uh_huh_tilt", "chicken_peck", "yeah_nod",
                "headbanger_combo", "side_peekaboo", "neck_recoil",
                "groovy_sway_and_roll", "sharp_side_tilt"
            ]
        elif "emotions" in dataset.lower():
            return [
                "amazed1", "anxiety1", "attentive1", "boredom1",
                "cheerful1", "confused1", "curious1", "enthusiastic1",
                "frustrated1", "happy1", "sad1", "surprised1",
                "thoughtful1", "welcoming1", "yes1", "no1"
            ]
        return []

    def get_endpoint_info(self, endpoint: str) -> bool:
        """Check if endpoint exists.

        Args:
            endpoint: Endpoint path (e.g., "/api/tracking/enable")

        Returns:
            True if endpoint exists, False otherwise
        """
        try:
            response = requests.get(
                f"{self.base_url}{endpoint}",
                timeout=self.timeout
            )
            # 200 = exists, 404 = doesn't exist, 405 = wrong method but exists
            return response.status_code in [200, 405]
        except Exception:
            return False

    @property
    def is_connected(self) -> bool:
        """Check if daemon is currently connected.

        Returns:
            True if connected, False otherwise
        """
        return self._connected

    # =========================================================================
    # Daemon Lifecycle Management
    # =========================================================================

    def start_daemon(
        self,
        scene: str = "minimal",
        backend: str = "mujoco",
        port: int = 8100,
        venv_path: Optional[Path] = None
    ) -> bool:
        """Start daemon as subprocess.

        Args:
            scene: Scene to load (minimal, empty)
            backend: Backend to use (mujoco, dummy)
            port: Port to run on
            venv_path: Path to virtual environment (optional)

        Returns:
            True if daemon started successfully, False otherwise
        """
        if self._daemon_process and self._daemon_process.poll() is None:
            logger.warning("Daemon already running")
            return True

        try:
            # Build command
            cmd = [
                "mjpython" if backend == "mujoco" else "python",
                "-m", "reachy_mini.daemon.app.main",
                "--sim",
                "--scene", scene,
                "--fastapi-port", str(port)
            ]

            # Activate venv if provided
            if venv_path:
                activate_script = venv_path / "bin" / "activate"
                cmd = [
                    "bash", "-c",
                    f"source {activate_script} && {' '.join(cmd)}"
                ]

            logger.info(f"Starting daemon: {' '.join(cmd)}")

            # Start process
            self._daemon_process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                universal_newlines=True
            )

            # Wait for daemon to start (up to 10 seconds)
            for i in range(20):
                time.sleep(0.5)
                if self.check_connection():
                    logger.info("Daemon started successfully")
                    return True

                # Check if process died
                if self._daemon_process.poll() is not None:
                    logger.error("Daemon process exited immediately")
                    return False

            logger.error("Daemon failed to start within 10 seconds")
            return False

        except Exception as e:
            logger.error(f"Failed to start daemon: {e}")
            return False

    def stop_daemon(self) -> bool:
        """Stop daemon subprocess.

        Returns:
            True if daemon stopped successfully, False otherwise
        """
        if not self._daemon_process:
            logger.debug("No daemon process to stop")
            return True

        try:
            logger.info("Stopping daemon...")

            # Try graceful shutdown first
            self._daemon_process.terminate()

            try:
                self._daemon_process.wait(timeout=5)
                logger.info("Daemon stopped gracefully")
            except subprocess.TimeoutExpired:
                logger.warning("Daemon didn't stop gracefully, killing...")
                self._daemon_process.kill()
                self._daemon_process.wait()
                logger.info("Daemon killed")

            self._daemon_process = None
            self._connected = False
            return True

        except Exception as e:
            logger.error(f"Failed to stop daemon: {e}")
            return False

    def is_daemon_running(self) -> bool:
        """Check if daemon subprocess is running.

        Returns:
            True if daemon process is running, False otherwise
        """
        if not self._daemon_process:
            return False

        return self._daemon_process.poll() is None

    def get_daemon_logs(self, lines: int = 50) -> Tuple[str, str]:
        """Get daemon stdout and stderr logs.

        Args:
            lines: Number of lines to retrieve

        Returns:
            Tuple of (stdout, stderr)
        """
        if not self._daemon_process:
            return ("", "No daemon process running")

        try:
            # This is a simplified version - real implementation would need
            # to read from pipes asynchronously to avoid blocking
            return ("Logs not yet implemented", "")
        except Exception as e:
            return ("", f"Error reading logs: {e}")
