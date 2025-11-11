"""Extension discovery and lifecycle management for Reachy Control Center.

Scans extension directories, loads manifests, and manages subprocess extensions.
"""

import json
import logging
import subprocess
import threading
import time
from pathlib import Path
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class Extension:
    """Represents a loaded extension."""

    name: str
    path: Path
    manifest: Dict[str, Any]
    is_builtin: bool = False
    enabled: bool = True
    process: Optional[subprocess.Popen] = None
    health_check_thread: Optional[threading.Thread] = None
    _stop_health_check: threading.Event = field(default_factory=threading.Event)

    @property
    def api_base_url(self) -> str:
        """Get extension's API base URL."""
        return self.manifest.get("extension", {}).get("api_base_url", "")

    @property
    def has_display(self) -> bool:
        """Check if extension has a display component."""
        display = self.manifest.get("display", {})
        return display.get("enabled", False)

    @property
    def display_url(self) -> str:
        """Get full display URL."""
        if not self.has_display:
            return ""
        display = self.manifest.get("display", {})
        endpoint = display.get("url", "/display")
        return f"{self.api_base_url}{endpoint}"

    @property
    def requires_daemon(self) -> bool:
        """Check if extension requires daemon to be running."""
        return self.manifest.get("extension", {}).get("requires_daemon", False)

    def get_required_endpoints(self) -> List[str]:
        """Get list of required daemon endpoints."""
        return self.manifest.get("extension", {}).get("requires_daemon_endpoints", [])


class ExtensionManager:
    """Manages extension discovery, loading, and lifecycle."""

    def __init__(
        self,
        extensions_dir: Path,
        builtin_dir: Optional[Path] = None,
        daemon_client=None
    ):
        """Initialize extension manager.

        Args:
            extensions_dir: Path to user extensions directory
            builtin_dir: Path to built-in extensions directory (optional)
            daemon_client: DaemonClient instance for endpoint validation
        """
        self.extensions_dir = Path(extensions_dir)
        self.builtin_dir = Path(builtin_dir) if builtin_dir else None
        self.daemon_client = daemon_client

        self.extensions: List[Extension] = []
        self.extension_by_name: Dict[str, Extension] = {}

        # Create extensions directory if it doesn't exist
        self.extensions_dir.mkdir(parents=True, exist_ok=True)
        if self.builtin_dir:
            self.builtin_dir.mkdir(parents=True, exist_ok=True)

    def discover_extensions(self) -> None:
        """Scan directories and load all extensions."""
        self.extensions.clear()
        self.extension_by_name.clear()

        logger.info("Discovering extensions...")

        # Scan built-in extensions first
        if self.builtin_dir and self.builtin_dir.exists():
            self._scan_directory(self.builtin_dir, is_builtin=True)

        # Scan user extensions
        if self.extensions_dir.exists():
            self._scan_directory(self.extensions_dir, is_builtin=False)

        logger.info(f"Found {len(self.extensions)} extensions ({sum(1 for e in self.extensions if e.is_builtin)} built-in)")

    def _scan_directory(self, directory: Path, is_builtin: bool) -> None:
        """Scan a directory for extensions.

        Args:
            directory: Directory to scan
            is_builtin: Whether these are built-in extensions
        """
        for ext_dir in directory.iterdir():
            if not ext_dir.is_dir():
                continue

            manifest_path = ext_dir / "manifest.json"
            if not manifest_path.exists():
                logger.debug(f"Skipping {ext_dir.name} (no manifest.json)")
                continue

            try:
                manifest = self._load_manifest(manifest_path)
                extension = Extension(
                    name=ext_dir.name,
                    path=ext_dir,
                    manifest=manifest,
                    is_builtin=is_builtin
                )

                # Validate extension
                if self._validate_extension(extension):
                    self.extensions.append(extension)
                    self.extension_by_name[extension.name] = extension
                    logger.info(f"Loaded extension: {extension.name} ({'builtin' if is_builtin else 'user'})")
                else:
                    logger.warning(f"Extension {extension.name} failed validation")

            except Exception as e:
                logger.error(f"Failed to load extension {ext_dir.name}: {e}")

    def _load_manifest(self, manifest_path: Path) -> Dict[str, Any]:
        """Load and parse manifest.json.

        Args:
            manifest_path: Path to manifest.json

        Returns:
            Parsed manifest dictionary
        """
        with open(manifest_path, 'r') as f:
            return json.load(f)

    def _validate_extension(self, extension: Extension) -> bool:
        """Validate extension manifest and requirements.

        Args:
            extension: Extension to validate

        Returns:
            True if extension is valid, False otherwise
        """
        # Check required manifest fields
        if "extension" not in extension.manifest:
            logger.error(f"{extension.name}: Missing 'extension' section in manifest")
            return False

        ext_info = extension.manifest["extension"]
        if "name" not in ext_info:
            logger.error(f"{extension.name}: Missing 'name' in extension section")
            return False

        # Check daemon requirements
        if extension.requires_daemon and self.daemon_client:
            required_endpoints = extension.get_required_endpoints()
            for endpoint in required_endpoints:
                if not self.daemon_client.get_endpoint_info(endpoint):
                    logger.warning(
                        f"{extension.name}: Required endpoint {endpoint} not available in daemon"
                    )
                    # Don't fail validation, just warn - daemon might add it later

        return True

    def get_enabled_extensions(self) -> List[Extension]:
        """Get list of enabled extensions.

        Returns:
            List of enabled Extension objects
        """
        return [ext for ext in self.extensions if ext.enabled]

    def get_extension(self, name: str) -> Optional[Extension]:
        """Get extension by name.

        Args:
            name: Extension name

        Returns:
            Extension object or None if not found
        """
        return self.extension_by_name.get(name)

    def start_extension(self, extension: Extension) -> bool:
        """Start an extension subprocess.

        Args:
            extension: Extension to start

        Returns:
            True if started successfully, False otherwise
        """
        if extension.is_builtin:
            logger.debug(f"{extension.name}: Built-in extension, no subprocess to start")
            return True

        if extension.process and extension.process.poll() is None:
            logger.warning(f"{extension.name}: Already running")
            return True

        lifecycle = extension.manifest.get("lifecycle", {})
        start_cmd = lifecycle.get("on_start", {})

        if not start_cmd:
            logger.warning(f"{extension.name}: No start command defined")
            return False

        try:
            # Parse command
            if isinstance(start_cmd, dict):
                command = start_cmd.get("command", "")
                env = start_cmd.get("environment", {})
            else:
                command = start_cmd
                env = {}

            logger.info(f"Starting extension {extension.name}: {command}")

            # Start subprocess
            extension.process = subprocess.Popen(
                command,
                shell=True,
                cwd=str(extension.path),
                env={**subprocess.os.environ, **env},
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE
            )

            # Wait a moment for process to start
            time.sleep(1)

            # Check if process is still running
            if extension.process.poll() is not None:
                logger.error(f"{extension.name}: Process exited immediately")
                return False

            # Start health check thread
            self._start_health_check(extension)

            logger.info(f"{extension.name}: Started successfully (PID {extension.process.pid})")
            return True

        except Exception as e:
            logger.error(f"Failed to start {extension.name}: {e}")
            return False

    def stop_extension(self, extension: Extension) -> None:
        """Stop an extension subprocess.

        Args:
            extension: Extension to stop
        """
        if extension.is_builtin:
            return

        # Stop health check thread
        if extension.health_check_thread:
            extension._stop_health_check.set()
            extension.health_check_thread.join(timeout=2)
            extension.health_check_thread = None

        if not extension.process:
            return

        # Try graceful shutdown via API
        lifecycle = extension.manifest.get("lifecycle", {})
        shutdown_endpoint = lifecycle.get("on_stop", {})

        if shutdown_endpoint:
            try:
                import requests
                if isinstance(shutdown_endpoint, dict):
                    endpoint = shutdown_endpoint.get("endpoint", "")
                    method = shutdown_endpoint.get("method", "POST")
                else:
                    endpoint = shutdown_endpoint
                    method = "POST"

                url = f"{extension.api_base_url}{endpoint}"
                requests.request(method, url, timeout=2)
                time.sleep(1)
            except Exception as e:
                logger.debug(f"Failed to call shutdown endpoint: {e}")

        # Force kill if still running
        if extension.process.poll() is None:
            logger.info(f"Terminating {extension.name} (PID {extension.process.pid})")
            extension.process.terminate()
            try:
                extension.process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                logger.warning(f"Killing {extension.name} (PID {extension.process.pid})")
                extension.process.kill()

        extension.process = None

    def _start_health_check(self, extension: Extension) -> None:
        """Start health check thread for extension.

        Args:
            extension: Extension to monitor
        """
        lifecycle = extension.manifest.get("lifecycle", {})
        healthcheck = lifecycle.get("healthcheck", {})

        if not healthcheck:
            return

        endpoint = healthcheck.get("endpoint", "/health")
        interval_ms = healthcheck.get("interval_ms", 5000)

        def health_check_loop():
            import requests
            while not extension._stop_health_check.is_set():
                try:
                    url = f"{extension.api_base_url}{endpoint}"
                    response = requests.get(url, timeout=2)
                    if response.status_code != 200:
                        logger.warning(f"{extension.name}: Health check failed (status {response.status_code})")
                except Exception as e:
                    logger.debug(f"{extension.name}: Health check error: {e}")

                extension._stop_health_check.wait(timeout=interval_ms / 1000.0)

        extension._stop_health_check.clear()
        extension.health_check_thread = threading.Thread(
            target=health_check_loop,
            daemon=True,
            name=f"HealthCheck-{extension.name}"
        )
        extension.health_check_thread.start()

    def stop_all_extensions(self) -> None:
        """Stop all running extensions."""
        logger.info("Stopping all extensions...")
        for extension in self.extensions:
            if extension.process:
                self.stop_extension(extension)
