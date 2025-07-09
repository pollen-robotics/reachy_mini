import asyncio
import platform
import subprocess
import sys
import threading
import time
from datetime import datetime
from importlib.metadata import PackageNotFoundError, distribution
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple, Union

# Platform detection
IS_WINDOWS = sys.platform == "win32"
IS_MACOS = sys.platform == "darwin"
IS_LINUX = sys.platform.startswith("linux")

# Global log storage and callbacks
_log_callbacks: List[Callable] = []
_process_logs: Dict[str, List[dict]] = {}


def get_package_entrypoints(package_name: str) -> List[Tuple[str, str, str]]:
    """
    Return a list of entry points for the given package.
    Each entry is a tuple: (group, name, value).
    Raises:
    - ValueError if the package isn't installed
    """
    try:
        dist = distribution(package_name)
    except PackageNotFoundError:
        raise ValueError(f"Package {package_name!r} not found")

    return [(ep.group, ep.name, ep.value) for ep in dist.entry_points]


def get_platform_info():
    """Get detailed platform information"""
    return {
        "system": platform.system(),
        "release": platform.release(),
        "version": platform.version(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "python_version": sys.version,
        "is_windows": IS_WINDOWS,
        "is_macos": IS_MACOS,
        "is_linux": IS_LINUX,
    }


def register_log_callback(callback: Callable):
    """Register a callback to receive log messages"""
    _log_callbacks.append(callback)


def unregister_log_callback(callback: Callable):
    """Unregister a log callback"""
    if callback in _log_callbacks:
        _log_callbacks.remove(callback)


def _broadcast_log(process_id: str, log_entry: dict):
    """Broadcast log entry to all registered callbacks"""
    if process_id not in _process_logs:
        _process_logs[process_id] = []

    _process_logs[process_id].append(log_entry)

    # Keep only last 1000 log entries per process
    if len(_process_logs[process_id]) > 1000:
        _process_logs[process_id] = _process_logs[process_id][-1000:]

    for callback in _log_callbacks:
        try:
            callback(process_id, log_entry)
        except Exception as e:
            print(f"Error in log callback: {e}")


def get_process_logs(process_id: str) -> List[dict]:
    """Get all logs for a specific process"""
    return _process_logs.get(process_id, [])


def clear_process_logs(process_id: str):
    """Clear logs for a specific process"""
    if process_id in _process_logs:
        del _process_logs[process_id]


class SubprocessHelper:
    """Enhanced subprocess helper with live logging"""

    def __init__(self, process_id: str, description: str = ""):
        self.process_id = process_id
        self.description = description
        self.process: Optional[subprocess.Popen] = None
        self.stdout_thread: Optional[threading.Thread] = None
        self.stderr_thread: Optional[threading.Thread] = None
        self._terminated = False

    def _log(self, level: str, message: str, stream: str = "info"):
        """Log a message with timestamp"""
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "level": level,
            "message": message,
            "stream": stream,
            "process_id": self.process_id,
            "description": self.description,
        }
        _broadcast_log(self.process_id, log_entry)

    def _stream_output(self, pipe, stream_name: str):
        """Stream output from a pipe to log callbacks"""
        try:
            for line in iter(pipe.readline, ""):
                if self._terminated:
                    break
                if line.strip():
                    self._log("info", line.strip(), stream_name)
            pipe.close()
        except Exception as e:
            if not self._terminated:
                self._log("error", f"Error reading {stream_name}: {e}", stream_name)

    def run_sync(
        self,
        cmd: List[str],
        cwd: Optional[Union[str, Path]] = None,
        timeout: Optional[int] = 300,
        env: Optional[dict] = None,
    ) -> subprocess.CompletedProcess:
        """
        Run a command synchronously with live logging
        Suitable for installation, updates, short-running tasks
        """
        cmd_str = " ".join(cmd)
        self._log("info", f"Starting command: {cmd_str}")
        if cwd:
            self._log("info", f"Working directory: {cwd}")

        try:
            # Clear previous logs for this process
            clear_process_logs(self.process_id)

            self.process = subprocess.Popen(
                cmd,
                cwd=str(cwd) if cwd else None,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                bufsize=1,
                universal_newlines=True,
                env=env,
            )

            # Start output streaming threads
            self.stdout_thread = threading.Thread(
                target=self._stream_output, args=(self.process.stdout, "stdout")
            )
            self.stderr_thread = threading.Thread(
                target=self._stream_output, args=(self.process.stderr, "stderr")
            )

            self.stdout_thread.start()
            self.stderr_thread.start()

            # Wait for process to complete
            try:
                exit_code = self.process.wait(timeout=timeout)
            except subprocess.TimeoutExpired:
                self._log("error", f"Command timed out after {timeout}s")
                self.terminate()
                raise Exception(f"Command timed out after {timeout}s: {cmd_str}")

            # Wait for output threads to finish
            if self.stdout_thread and self.stdout_thread.is_alive():
                self.stdout_thread.join(timeout=5)
            if self.stderr_thread and self.stderr_thread.is_alive():
                self.stderr_thread.join(timeout=5)

            self._log("info", f"Command completed with exit code: {exit_code}")

            if exit_code != 0:
                error_msg = f"Command failed with exit code {exit_code}: {cmd_str}"
                self._log("error", error_msg)
                # Create a mock CompletedProcess for compatibility
                result = subprocess.CompletedProcess(
                    args=cmd, returncode=exit_code, stdout="", stderr=""
                )
                raise subprocess.CalledProcessError(exit_code, cmd_str)

            # Create a mock CompletedProcess for compatibility
            return subprocess.CompletedProcess(
                args=cmd, returncode=exit_code, stdout="", stderr=""
            )

        except Exception as e:
            self._log("error", f"Command execution failed: {e}")
            raise

    def run_async(
        self,
        cmd: List[str],
        cwd: Optional[Union[str, Path]] = None,
        env: Optional[dict] = None,
    ) -> subprocess.Popen:
        """
        Run a command asynchronously with live logging
        Suitable for long-running apps
        Returns the Popen object for external management
        """
        cmd_str = " ".join(cmd)
        self._log("info", f"Starting async command: {cmd_str}")
        if cwd:
            self._log("info", f"Working directory: {cwd}")

        try:
            # Clear previous logs for this process
            clear_process_logs(self.process_id)

            self.process = subprocess.Popen(
                cmd,
                cwd=str(cwd) if cwd else None,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                bufsize=1,
                universal_newlines=True,
                env=env,
            )

            # Start output streaming threads
            self.stdout_thread = threading.Thread(
                target=self._stream_output, args=(self.process.stdout, "stdout")
            )
            self.stderr_thread = threading.Thread(
                target=self._stream_output, args=(self.process.stderr, "stderr")
            )

            self.stdout_thread.daemon = True  # Don't block program exit
            self.stderr_thread.daemon = True

            self.stdout_thread.start()
            self.stderr_thread.start()

            return self.process

        except Exception as e:
            self._log("error", f"Failed to start async command: {e}")
            raise

    def terminate(self):
        """Terminate the running process"""
        self._terminated = True
        if self.process:
            try:
                self._log("info", "Terminating process...")
                self.process.terminate()
                try:
                    self.process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    self._log(
                        "warning", "Process didn't terminate gracefully, killing..."
                    )
                    self.process.kill()
                    self.process.wait()
                self._log("info", "Process terminated")
            except Exception as e:
                self._log("error", f"Error terminating process: {e}")

    def is_running(self) -> bool:
        """Check if the process is still running"""
        return self.process is not None and self.process.poll() is None


# Convenience functions for backward compatibility
def run_subprocess(
    cmd: List[str],
    cwd: Optional[str] = None,
    timeout: int = 300,
    process_id: Optional[str] = None,
    description: str = "",
) -> subprocess.CompletedProcess:
    """
    Run subprocess with live logging (sync version)
    Backward compatible with existing code
    """
    if process_id is None:
        process_id = f"subprocess_{int(time.time())}"

    helper = SubprocessHelper(process_id, description)
    return helper.run_sync(cmd, cwd, timeout)


def run_subprocess_async(
    cmd: List[str],
    cwd: Optional[str] = None,
    env: Optional[dict] = None,
    process_id: Optional[str] = None,
    description: str = "",
) -> tuple[subprocess.Popen, SubprocessHelper]:
    """
    Run subprocess with live logging (async version)
    Returns both the process and the helper for management
    """
    if process_id is None:
        process_id = f"subprocess_async_{int(time.time())}"

    helper = SubprocessHelper(process_id, description)
    process = helper.run_async(cmd, cwd, env)
    return process, helper
