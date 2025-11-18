"""Reachy Mini Application Base Class.

This module provides a base class for creating Reachy Mini applications.
It includes methods for running the application, stopping it gracefully,
and creating a new app project with a specified name and path.

It uses Jinja2 templates to generate the necessary files for the app project.
"""

import threading
import traceback
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict

from jinja2 import Environment, FileSystemLoader

from reachy_mini.reachy_mini import ReachyMini


class ReachyMiniApp(ABC):
    """Base class for Reachy Mini applications."""

    def __init__(self) -> None:
        """Initialize the Reachy Mini app."""
        self.stop_event = threading.Event()
        self.error: str = ""

    def wrapped_run(self, *args: Any, **kwargs: Any) -> None:
        """Wrap the run method with Reachy Mini context management."""
        try:
            with ReachyMini(*args, **kwargs) as reachy_mini:
                self.run(reachy_mini, self.stop_event)
        except Exception:
            self.error = traceback.format_exc()
            raise

    @abstractmethod
    def run(self, reachy_mini: ReachyMini, stop_event: threading.Event) -> None:
        """Run the main logic of the app.

        Args:
            reachy_mini (ReachyMini): The Reachy Mini instance to interact with.
            stop_event (threading.Event): An event that can be set to stop the app gracefully.

        """
        pass

    def stop(self) -> None:
        """Stop the app gracefully."""
        self.stop_event.set()
        print("App is stopping...")


def make_app_project(app_name: str, path: Path) -> None:
    """Create a new Reachy Mini app project with the given name at the specified path.

    Args:
        app_name (str): The name of the app to create.
        path (Path): The directory where the app project will be created.

    """
    TEMPLATE_DIR = Path(__file__).parent / "templates"
    env = Environment(loader=FileSystemLoader(TEMPLATE_DIR))

    def render_template(filename: str, context: Dict[str, str]) -> str:
        template = env.get_template(filename)
        return template.render(context)

    base_path = Path(path).resolve() / app_name
    if base_path.exists():
        print(f"❌ Folder {base_path} already exists.")
        return

    module_name = app_name.replace("-", "_")
    class_name = "".join(word.capitalize() for word in module_name.split("_"))

    base_path.mkdir()
    (base_path / module_name).mkdir()

    # Generate files
    context = {
        "app_name": app_name,
        "package_name": app_name,
        "module_name": module_name,
        "class_name": class_name,
    }

    (base_path / module_name / "__init__.py").touch()
    (base_path / module_name / "main.py").write_text(
        render_template("main.py.j2", context)
    )
    (base_path / "pyproject.toml").write_text(
        render_template("pyproject.toml.j2", context)
    )
    (base_path / "README.md").write_text(render_template("README.md.j2", context))

    print(f"✅ Created app in {base_path}/")


def main() -> None:
    """Run the command line interface to create a new Reachy Mini app project."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Create a new Reachy Mini app project."
    )
    parser.add_argument("app_name", type=str, help="Name of the app to create.")
    parser.add_argument(
        "--path",
        type=Path,
        default=Path.cwd(),
        help="Path where the app project will be created.",
    )

    args = parser.parse_args()
    make_app_project(args.app_name, args.path)


if __name__ == "__main__":
    main()
