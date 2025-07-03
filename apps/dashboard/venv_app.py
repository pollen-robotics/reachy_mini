import json
import os
import shutil
import subprocess
import venv
from pathlib import Path
from typing import List, Optional, Tuple

from utils import (
    IS_WINDOWS,
    SubprocessHelper,
    run_subprocess_async,
)


class VenvAppManager:
    """Manages apps installed in virtual environments"""

    def __init__(self, apps_dir: Path):
        self.apps_dir = apps_dir
        self.dashboard_dir = apps_dir.parent

    def create_venv(self, app_name: str) -> Path:
        try:
            """Create a virtual environment for the app"""
            venv_path = (self.apps_dir / app_name / "venv").absolute()
            venv_path.parent.mkdir(parents=True, exist_ok=True)
            if venv_path.exists():
                self.remove_directory_robust(venv_path)

            # print("Creating venv using venv module...")
            # Create venv with absolute paths
            venv.create(
                str(venv_path), with_pip=True, clear=True, symlinks=not IS_WINDOWS
            )
            # print(f"Venv created successfully at {venv_path}")
            # Verify the venv was created successfully
            python_path = self.get_venv_python_from_path(venv_path)
            if not python_path.exists():
                raise Exception(
                    f"Python executable not found after venv creation: {python_path}"
                )

            return venv_path
        except Exception as e:
            print(f"Error creating venv for {app_name}: {e}")
            raise

    def get_venv_python_from_path(self, venv_path: Path) -> Path:
        """Get python executable path from venv path"""
        if IS_WINDOWS:
            python_exe = venv_path / "Scripts" / "python.exe"
        else:
            python_exe = venv_path / "bin" / "python"
        return python_exe.absolute()

    def get_venv_python(self, app_name: str) -> Path:
        """Get path to python executable in the app's venv"""
        venv_path = self.apps_dir / app_name / "venv"
        return self.get_venv_python_from_path(venv_path)

    def get_entrypoint(self, app_name: str) -> Optional[Path]:
        return self.apps_dir / app_name / "venv" / "bin" / app_name

    def get_venv_pip(self, app_name: str) -> Path:
        """Get path to pip executable in the app's venv"""
        venv_path = self.apps_dir / app_name / "venv"

        if IS_WINDOWS:
            # Try different possible pip locations on Windows
            possible_paths = [
                venv_path / "Scripts" / "pip.exe",
                venv_path / "Scripts" / "pip3.exe",
                venv_path / "Scripts" / "pip",
            ]
        else:
            # Try different possible pip locations on Unix
            possible_paths = [venv_path / "bin" / "pip", venv_path / "bin" / "pip3"]

        # Return the first existing path
        for path in possible_paths:
            if path.exists():
                return path.absolute()

        # If no pip found, return the most likely path (it will be created during pip upgrade)
        return possible_paths[0].absolute()

    def remove_directory_robust(self, path: Path) -> bool:
        """Robust directory removal that works on all platforms"""
        if not path.exists():
            return True

        for attempt in range(3):
            try:
                if IS_WINDOWS:
                    import stat

                    def handle_remove_readonly(func, path, exc):
                        try:
                            os.chmod(path, stat.S_IWRITE)
                            func(path)
                        except Exception:
                            pass

                    shutil.rmtree(path, onerror=handle_remove_readonly)
                else:
                    shutil.rmtree(path)

                if not path.exists():
                    return True

            except Exception:
                # Wait before retry
                import time

                time.sleep(0.5)

                # Try alternative methods
                try:
                    if IS_WINDOWS:
                        subprocess.run(
                            ["cmd", "/c", "rd", "/s", "/q", str(path)],
                            capture_output=True,
                            text=True,
                        )
                    else:
                        subprocess.run(
                            ["rm", "-rf", str(path)], capture_output=True, text=True
                        )
                except Exception:
                    pass

        return not path.exists()

    # def run_app_in_venv(self, app_name: str) -> subprocess.Popen:
    #     """Run an app in its virtual environment (legacy method)"""
    #     print("Running app with legacy method")
    #     python_path = self.get_venv_python(app_name)
    #     app_dir = (self.apps_dir / app_name).absolute()

    #     if not python_path.exists():
    #         raise Exception(f"Python executable not found: {python_path}")

    #     metadata = self.get_app_metadata(app_name)
    #     package_name = metadata.get("package_name", app_name)
    #     print(f"Running {app_name} with package name {package_name}")
    #     print(f"App directory: {app_dir}")

    #     app_entrypoint = (
    #         self.get_package_location(python_path, package_name) / "main.py"
    #     )

    #     return subprocess.Popen(
    #         [str(python_path), str(app_entrypoint)],
    #         cwd=str(app_dir),
    #         env=self._get_app_env(app_name),
    #     )

    def run_app_in_venv_with_logging(
        self, app_name: str
    ) -> Tuple[subprocess.Popen, SubprocessHelper]:
        """Run an app in its virtual environment with live logging"""
        print("Running app with enhanced logging")
        python_path = self.get_venv_python(app_name)
        app_dir = (self.apps_dir / app_name).absolute()

        if not python_path.exists():
            raise Exception(f"Python executable not found: {python_path}")

        metadata = self.get_app_metadata(app_name)
        package_name = metadata.get("package_name", app_name)
        print(f"Running {app_name} with package name {package_name}")
        print(f"App directory: {app_dir}")

        # app_entrypoint = (
        #     self.get_package_location(python_path, package_name) / f"{package_name}"
        # )

        # if not app_entrypoint.exists():
        #     raise Exception(f"App entrypoint not found: {app_entrypoint}")
        app_entrypoint = self.get_entrypoint(app_name)
        print(f"App entrypoint: {app_entrypoint}")

        # Use the enhanced subprocess helper
        process_id = f"app_{app_name}"
        description = f"Running {app_name}"
        # print(get_package_entrypoints(package_name))
        print(f"Entrypoint: {app_entrypoint}")
        print("xxx")

        process, helper = run_subprocess_async(
            cmd=[str(app_entrypoint)],
            cwd=str(app_dir),
            env=self._get_app_env(app_name),
            process_id=process_id,
            description=description,
        )

        return process, helper

    def _get_app_env(self, app_name: str) -> dict:
        """Get environment variables for running the app"""
        env = os.environ.copy()

        venv_path = (self.apps_dir / app_name / "venv").absolute()

        if IS_WINDOWS:
            scripts_dir = venv_path / "Scripts"
            env["PATH"] = f"{scripts_dir}{os.pathsep}{env.get('PATH', '')}"
        else:
            bin_dir = venv_path / "bin"
            env["PATH"] = f"{bin_dir}{os.pathsep}{env.get('PATH', '')}"

        env["VIRTUAL_ENV"] = str(venv_path)

        # Add app directory to PYTHONPATH
        app_dir = self.apps_dir / app_name
        python_paths = [str(app_dir.absolute())]

        # Add src directory if it exists
        src_dir = app_dir / "src"
        if src_dir.exists():
            python_paths.append(str(src_dir.absolute()))

        env["PYTHONPATH"] = os.pathsep.join(python_paths)
        env["PYTHONNOUSERSITE"] = "1"
        env.pop("PYTHONHOME", None)

        return env

    def list_installed_apps(self) -> List[dict]:
        """List all installed apps with metadata"""
        apps = []
        if not self.apps_dir.exists():
            return apps

        for app_dir in self.apps_dir.iterdir():
            if app_dir.is_dir() and (app_dir / "venv").exists():
                metadata = self.get_app_metadata(app_dir.name)
                apps.append(
                    {
                        "name": app_dir.name,
                        "path": str(app_dir),
                        "venv_path": str(app_dir / "venv"),
                        "installed_date": metadata.get("installed_date"),
                        "source_url": metadata.get("source_url"),
                        "last_updated": metadata.get("last_updated"),
                        "package_name": metadata.get("package_name"),
                    }
                )
        return apps

    def save_app_metadata(self, app_name: str, metadata: dict):
        """Save app metadata to a JSON file"""
        metadata_file = self.apps_dir / app_name / "app_metadata.json"
        try:
            metadata_file.parent.mkdir(parents=True, exist_ok=True)
            with open(metadata_file, "w") as f:
                json.dump(metadata, f, indent=2)
        except Exception as e:
            print(f"Warning: Could not save metadata for {app_name}: {e}")

    def get_app_metadata(self, app_name: str) -> dict:
        """Get app metadata from JSON file"""
        metadata_file = self.apps_dir / app_name / "app_metadata.json"
        try:
            if metadata_file.exists():
                with open(metadata_file, "r") as f:
                    return json.load(f)
        except Exception as e:
            print(f"Warning: Could not read metadata for {app_name}: {e}")
        return {}

    def remove_app_completely(self, app_name: str) -> bool:
        """Remove app and its virtual environment completely"""
        try:
            app_dir = self.apps_dir / app_name

            if not app_dir.exists():
                return True  # Already removed

            # Kill any running processes if on Windows
            if IS_WINDOWS:
                try:
                    subprocess.run(
                        ["taskkill", "/F", "/FI", f"WINDOWTITLE eq *{app_name}*"],
                        capture_output=True,
                    )
                except Exception:
                    pass

            # Remove the directory
            success = self.remove_directory_robust(app_dir)
            return success and not app_dir.exists()

        except Exception as e:
            print(f"Error during app removal {app_name}: {e}")
            return False

    def get_package_location(self, venv_exec, package_name: str) -> Path:
        """
        Return the installation directory for a given package by invoking `pip show`.
        Raises:
        - ValueError if the package isn't found
        - RuntimeError if the Location field is missing
        """
        try:
            result = subprocess.run(
                [venv_exec, "-m", "pip", "show", package_name],
                capture_output=True,
                text=True,
                check=True,
            )
        except subprocess.CalledProcessError:
            raise ValueError(f"Package {package_name!r} not found")

        for line in result.stdout.splitlines():
            if line.startswith("Location:"):
                _, loc = line.split(":", 1)
                return Path(loc.strip()) / package_name

        raise RuntimeError(
            f"Could not parse installation location for {package_name!r}"
        )
